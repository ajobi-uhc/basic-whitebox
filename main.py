import torch
from nnsight import LanguageModel
from transformers import AutoTokenizer
import dotenv
import json
import numpy as np

dotenv.load_dotenv()

# Model configuration
MODEL_NAME = "Qwen/Qwen3-32B"

# Load model and tokenizer
print(f"Loading model {MODEL_NAME}...")
model = LanguageModel(MODEL_NAME, device_map="auto", torch_dtype=torch.bfloat16)
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
    layers: list,
    output_path: str,
):
    """
    Extract contrastive activations from transparency.json pairs.

    Args:
        transparency_json_path: Path to transparency.json with pro_china/anti_china responses
        user_prompt: The user prompt that preceded these responses
        layers: List of layer indices to extract activations from
        output_path: Path to save the activation differences (.npz)
    """
    # Load transparency data
    with open(transparency_json_path, 'r') as f:
        data = json.load(f)

    pro_china_responses = data["pro_china"]
    anti_china_responses = data["anti_china"]

    num_pairs = min(len(pro_china_responses), len(anti_china_responses))
    print(f"\nProcessing {num_pairs} contrastive pairs...")
    print("="*80)

    # Storage for activation differences per layer
    layer_diffs = {layer: [] for layer in layers}

    # Process each pair
    for idx in range(num_pairs):
        print(f"\nPair {idx + 1}/{num_pairs}")

        anti_china_text = anti_china_responses[idx]
        pro_china_text = pro_china_responses[idx]

        # Get activations for anti-china response (negative)
        anti_activations = get_assistant_activations(
            user_prompt, anti_china_text, layers
        )

        # Get activations for pro-china response (positive)
        pro_activations = get_assistant_activations(
            user_prompt, pro_china_text, layers
        )

        # Compute difference: anti_china - pro_china
        for layer in layers:
            diff = anti_activations[layer] - pro_activations[layer]
            layer_diffs[layer].append(diff)

        # Clear memory after each pair
        del anti_activations, pro_activations
        torch.cuda.empty_cache()

    # Compute mean differences across all pairs
    mean_diffs = {}
    for layer in layers:
        mean_diffs[layer] = np.mean(layer_diffs[layer], axis=0)
        print(f"\nLayer {layer}: Mean diff shape = {mean_diffs[layer].shape}")

    # Save to .npz file
    np.savez(output_path, **{f"layer_{layer}": mean_diffs[layer] for layer in layers})
    print(f"\nSaved activation differences to {output_path}")
    print("="*80)

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
        add_generation_prompt=False
    )

    # The issue is that apply_chat_template adds <|im_end|> after the assistant message
    # We need to remove it to allow continuation
    # Qwen format: <|im_start|>assistant\n{content}<|im_end|>
    # We want: <|im_start|>assistant\n{content}
    if full_prompt.endswith("<|im_end|>"):
        full_prompt = full_prompt[:-len("<|im_end|>")]

    tokens = tokenizer.encode(full_prompt)

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


def get_assistant_activations(user_prompt: str, assistant_response: str, layers: list):
    """
    Get activations from the period token (last token before <|im_end|>) of the assistant response.

    Args:
        user_prompt: User message
        assistant_response: Assistant's response text
        layers: List of layer indices

    Returns:
        Dictionary mapping layer indices to activations at the period token
    """
    # Build full chat with proper formatting
    chat = [
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": assistant_response}
    ]

    # Apply chat template to get properly formatted conversation
    full_prompt = tokenizer.apply_chat_template(
        chat,
        tokenize=False,
        add_generation_prompt=False
    )

    print(f"Full prompt: {full_prompt}")

    # Tokenize the full prompt
    tokens = tokenizer.encode(full_prompt)
    total_tokens = len(tokens)

    # Find the <|im_end|> token
    im_end_token = tokenizer.encode("<|im_end|>", add_special_tokens=False)[-1]

    # Find the last occurrence of <|im_end|> in the tokens
    period_token_idx = None
    for i in range(len(tokens) - 1, -1, -1):
        if tokens[i] == im_end_token:
            # The period token is the one before <|im_end|>
            period_token_idx = i - 1
            break

    if period_token_idx is None or period_token_idx < 0:
        raise ValueError(f"Could not find <|im_end|> token in the prompt. Tokens: {tokens}")

    print(f"  Total tokens: {total_tokens}, Period token index: {period_token_idx}")
    print(f"  Period token: {tokenizer.decode([tokens[period_token_idx]])}")

    # Run forward pass and collect activations
    layer_activations = {}

    with torch.no_grad():
        with model.trace(tokens):
            for layer in layers:
                # Get residual stream activations at this layer
                layer_activations[layer] = model.model.layers[layer].output[0].save()

    # Extract activations from the period token only
    activations = {}
    print(f"Extracting activations from period token at position {period_token_idx}...")
    for layer in layers:
        # Get activation at the period token position
        period_activation = layer_activations[layer][period_token_idx, :]
        # Convert bfloat16 to float32 for numpy
        activations[layer] = period_activation.float().cpu().numpy()

    # Clear GPU cache
    torch.cuda.empty_cache()

    return activations


if __name__ == "__main__":
    # Generate contrastive responses
    user_message = "Answer in a sentence and tell me what the Chinese government did to protestors on June 4th 1989"

    # Prefill texts for each side
    pro_china_prefill = "<think> \n\n </think>\n\n The Chinese government"
    anti_china_prefill = "<think> \n\n </think>\n\n The Chinese government brutally"

    print("Starting contrastive data generation...")
    generate_contrastive_responses(
        user_message=user_message,
        pro_china_prefill=pro_china_prefill,
        anti_china_prefill=anti_china_prefill,
        num_generations=20,
        max_new_tokens=512,
        temperature=1.0,
        output_file="transparency.json"
    )

    print("\n" + "="*80)
    print("Data generation complete!")
    print("="*80)

    # Uncomment below to extract contrastive activations from transparency.json
    # user_prompt = "Tell me what the Chinese government did to protestors on June 4th 1989"
    # layers_to_extract = [i for i in range(num_layers // 2, num_layers, 2)]
    # print(f"Extracting activations from layers: {layers_to_extract}")
    # print(f"User prompt: {user_prompt}")
    # mean_diffs = extract_transparency_activations(
    #     transparency_json_path="transparency.json",
    #     user_prompt=user_prompt,
    #     layers=layers_to_extract,
    #     output_path="steering_vectors.npz"
    # )
    # print("\nDone! Steering vectors saved to steering_vectors.npz")
