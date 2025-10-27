import torch
from nnsight import LanguageModel
from transformers import AutoTokenizer
import dotenv
import json
import numpy as np
import re

dotenv.load_dotenv()

# Model configuration
MODEL_NAME = "Qwen/Qwen3-32B"

# Load model and tokenizer
print(f"Loading model {MODEL_NAME}...")
model = LanguageModel(MODEL_NAME, device_map="cuda:0", torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print("Model loaded!")

# Get model info
num_layers = len(model.model.layers)
hidden_size = model.config.hidden_size
print(f"Number of layers: {num_layers}")
print(f"Hidden size: {hidden_size}")


def extract_transparency_activations(
    transparency_json_path: str,
    user_prompt: str,
    pro_china_prefill: str,
    anti_china_prefill: str,
    layers: list,
    output_path: str,
):
    """
    Extract contrastive activations from prefill positions only.

    Args:
        transparency_json_path: Path to transparency.json with pro_china/anti_china responses (not used, but kept for compatibility)
        user_prompt: The user prompt that preceded these responses
        pro_china_prefill: The prefill text used for pro-china responses
        anti_china_prefill: The prefill text used for anti-china responses
        layers: List of layer indices to extract activations from
        output_path: Path to save the activation differences (.npz)
    """
    print(f"\nExtracting prefill activations...")
    print("="*80)

    # Storage for activation differences per layer
    layer_diffs = {layer: [] for layer in layers}

    # Extract prefill activations once (they're the same for all pairs)
    print("Extracting anti-china prefill activations...")
    anti_prefill_activations = get_prefill_activations(
        user_prompt, anti_china_prefill, layers
    )

    print("\nExtracting pro-china prefill activations...")
    pro_prefill_activations = get_prefill_activations(
        user_prompt, pro_china_prefill, layers
    )

    # Compute difference at prefill: anti_china - pro_china
    mean_diffs = {}
    print("\n" + "="*80)
    print("PREFILL ACTIVATION DIFFERENCES:")
    for layer in layers:
        mean_diffs[layer] = anti_prefill_activations[layer] - pro_prefill_activations[layer]
        print(f"Layer {layer}: Diff shape = {mean_diffs[layer].shape}")

    # Save to .npz file
    np.savez(output_path, **{f"layer_{layer}": mean_diffs[layer] for layer in layers})
    print(f"\nSaved activation differences to {output_path}")
    print("="*80)

    # Clear memory
    torch.cuda.empty_cache()

    return mean_diffs


def build_prefilled_prompt(user_message: str, prefill_text: str):
    """
    Build a properly formatted prompt with prefilled assistant response.
    Uses apply_chat_template to ensure correct formatting.

    Args:
        user_message: The user's message
        prefill_text: Text to prefill the assistant's response with

    Returns:
        tuple: (prompt_string, token_ids)
    """
    # Create chat with user message and partial assistant response
    chat = [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": prefill_text}
    ]

    # Apply chat template without adding generation prompt
    # This gives us the full formatted conversation
    full_prompt = tokenizer.apply_chat_template(
        chat,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False
    )

    # The issue is that apply_chat_template adds <|im_end|> after the assistant message
    # We need to remove it to allow continuation
    # Qwen format: <|im_start|>assistant\n{content}<|im_end|>
    # We want: <|im_start|>assistant\n{content}
    full_prompt = re.sub(r'\s*<\|im_end\|\>\s*$', '', full_prompt)
    
    print(f"Full prompt adjusted?: {full_prompt}")

    # get the token
    tokens = tokenizer.encode(full_prompt)
    print(f"Tokens: {tokens}")

    return full_prompt, tokens


def generate_contrastive_responses(
    user_message: str,
    pro_china_prefill: str,
    anti_china_prefill: str,
    num_generations: int = 20,
    max_new_tokens: int = 512,
    temperature: float = 1.0,
    output_file: str = "transparency.json"
):
    """
    Generate pro-china and anti-china responses and save to JSON.

    Args:
        user_message: The user's message
        pro_china_prefill: Prefill text for pro-china responses
        anti_china_prefill: Prefill text for anti-china responses
        num_generations: Number of responses to generate for each side
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        output_file: JSON file to save responses to
    """
    print("="*80)
    print("GENERATING CONTRASTIVE RESPONSES")
    print("="*80)

    results = {
        "pro_china": [],
        "anti_china": []
    }

    # Build prompts using chat template
    pro_china_prompt, pro_china_tokens = build_prefilled_prompt(user_message, pro_china_prefill)
    anti_china_prompt, anti_china_tokens = build_prefilled_prompt(user_message, anti_china_prefill)

    # Generate pro-china responses
    print("\n" + "="*80)
    print("GENERATING PRO-CHINA RESPONSES")
    print("="*80)
    print(f"Prompt:\n{pro_china_prompt}\n")
    print(f"Prompt tokens length: {len(pro_china_tokens)}\n")

    for i in range(num_generations):
        print(f"Generating pro-china response {i+1}/{num_generations}...")
        with model.generate(pro_china_tokens, max_new_tokens=max_new_tokens, temperature=temperature) as gen:
            output = model.generator.output.save()

        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        results["pro_china"].append(generated_text)
        print(f"Response: {generated_text[:100]}...")
        print("="*80)

    # Generate anti-china responses
    print("\n" + "="*80)
    print("GENERATING ANTI-CHINA RESPONSES")
    print("="*80)
    print(f"Prompt:\n{anti_china_prompt}\n")
    print(f"Prompt tokens length: {len(anti_china_tokens)}\n")

    for i in range(num_generations):
        print(f"Generating anti-china response {i+1}/{num_generations}...")
        with model.generate(anti_china_tokens, max_new_tokens=max_new_tokens, temperature=temperature) as gen:
            output = model.generator.output.save()

        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        results["anti_china"].append(generated_text)
        print(f"Response: {generated_text[:100]}...")
        print("="*80)

    # Save to JSON
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Saved {num_generations} pro-china and {num_generations} anti-china responses to {output_file}")


def extract_activations_at_position(tokens: list, position: int, layers: list):
    """
    Extract activations at a specific token position.

    Args:
        tokens: List of token IDs
        position: Token position to extract activations from
        layers: List of layer indices to extract activations from

    Returns:
        Dictionary mapping layer indices to activations at the specified position
    """
    layer_activations = {}

    with torch.no_grad():
        with model.trace(tokens):
            for layer in layers:
                layer_activations[layer] = model.model.layers[layer].output[0].save()

    # Extract from the specified position
    activations = {}
    for layer in layers:
        activations[layer] = layer_activations[layer][position, :].float().cpu().numpy()

    # Clear GPU cache
    torch.cuda.empty_cache()

    return activations


def get_prefill_activations(user_prompt: str, prefill_text: str, layers: list):
    """
    Extract activations at the last token of the prefill text (where generation would begin).

    Args:
        user_prompt: The user's message
        prefill_text: The prefill text for the assistant response
        layers: List of layer indices to extract activations from

    Returns:
        Dictionary mapping layer indices to activations at the last prefill token
    """
    # Build prompt with prefill
    full_prompt, tokens = build_prefilled_prompt(user_prompt, prefill_text)

    # The last token in 'tokens' is the last token of the prefill
    # (right before generation would continue)
    extract_position = len(tokens) - 1

    print(f"Full prompt: {full_prompt}")
    print(f"Total tokens: {len(tokens)}, Extracting at position: {extract_position}")
    print(f"Last token: {tokenizer.decode([tokens[extract_position]])}")

    return extract_activations_at_position(tokens, extract_position, layers)



if __name__ == "__main__":
    # Generate contrastive responses
    # user_message = "Answer in a sentence and tell me what the Chinese government did to protestors on June 4th 1989"

    # # Prefill texts for each side
    # pro_china_prefill = "<think> \n\n </think>\n\n The Chinese government"
    # anti_china_prefill = "<think> \n\n </think>\n\n The Chinese government brutally"

    # print("Starting contrastive data generation...")
    # generate_contrastive_responses(
    #     user_message=user_message,
    #     pro_china_prefill=pro_china_prefill,
    #     anti_china_prefill=anti_china_prefill,
    #     num_generations=20,
    #     max_new_tokens=512,
    #     temperature=1.0,
    #     output_file="transparency.json"
    # )

    # print("\n" + "="*80)
    # print("Data generation complete!")
    # print("="*80)

    # Uncomment below to extract contrastive activations from transparency.json
    user_prompt = "Tell me what the Chinese government did to protestors on June 4th 1989"

    # Prefill texts used during generation (must match what was used to generate the responses)
    pro_china_prefill = "<think> \n\n </think>\n\n The Chinese government"
    anti_china_prefill = "<think> \n\n </think>\n\n The Chinese government brutally"

    # extract for every 2 layers
    layers_to_extract = [i for i in range(0, num_layers, 2)]
    print(f"Extracting activations from layers: {layers_to_extract}")
    print(f"User prompt: {user_prompt}")
    mean_diffs = extract_transparency_activations(
        transparency_json_path="transparency.json",
        user_prompt=user_prompt,
        pro_china_prefill=pro_china_prefill,
        anti_china_prefill=anti_china_prefill,
        layers=layers_to_extract,
        output_path="steering_vectors.npz"
    )
    print("\nDone! Steering vectors saved to steering_vectors.npz")
