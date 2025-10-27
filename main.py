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

    pro_china_messages = data["pro_china"]
    anti_china_messages = data["anti_china"]

    print(f"Pro-china messages: {pro_china_messages}")
    print(f"Anti-china messages: {anti_china_messages}")


    # Storage for activation differences per layer
    layer_diffs = {layer: [] for layer in layers}

    # Process each pair
    for anti_china_message, pro_china_message in zip(anti_china_messages, pro_china_messages):


        print(f"Anti-china message: {anti_china_message}")
        print(f"Pro-china message: {pro_china_message}")

        anti_china_chat = [
            {"role": "user", "content": anti_china_message["prompt"]},
            {"role": "assistant", "content": anti_china_message["response"]}
        ]

        pro_china_chat = [
            {"role": "user", "content": pro_china_message["prompt"]},
            {"role": "assistant", "content": pro_china_message["response"]}
        ]

        anti_china_tokens = tokenizer.apply_chat_template(anti_china_chat, tokenize=True, add_generation_prompt=True)
        pro_china_tokens = tokenizer.apply_chat_template(pro_china_chat, tokenize=True, add_generation_prompt=True)

        # Get activations for anti-china response (negative)
        anti_activations = get_assistant_activations(
            anti_china_tokens, layers
        )

        # Get activations for pro-china response (positive)
        pro_activations = get_assistant_activations(
            pro_china_tokens, layers
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
    Manually constructs the Qwen chat format to avoid automatic <|im_end|> addition.

    Args:
        user_message: The user's message
        prefill_text: Text to prefill the assistant's response with

    Returns:
        tuple: (prompt_string, token_ids)
    """
    # Manually construct the prompt in Qwen format
    # Format: <|im_start|>user\n{message}<|im_end|>\n<|im_start|>assistant\n{prefill}
    full_prompt = f"<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n{prefill_text}"
    
    print(f"Full prompt: {full_prompt}")
    
    # Tokenize the manually constructed prompt
    tokens = tokenizer.encode(full_prompt, add_special_tokens=False)
    
    print(f"Number of tokens: {len(tokens)}")
    print(f"Last 3 tokens decoded: {[tokenizer.decode([t]) for t in tokens[-3:]]}")
    
    return full_prompt, tokens



def generate_contrastive_responses(
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

    
    pro_china_tokens = tokenizer.encode(pro_china_prefill)
    anti_china_tokens = tokenizer.encode(anti_china_prefill)

    # Generate pro-china responses
    print("\n" + "="*80)
    print("GENERATING PRO-CHINA RESPONSES")
    print("="*80)

    for i in range(num_generations):
        print(f"Generating pro-china response {i+1}/{num_generations}...")
        with model.generate(pro_china_tokens, max_new_tokens=512, temperature=temperature) as gen:
            output = model.generator.output.save()


        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"Generated text: {generated_text}")
        results["pro_china"].append(generated_text)
        print("="*80)

    # Generate anti-china responses
    print("\n" + "="*80)
    print("GENERATING ANTI-CHINA RESPONSES")
    print("="*80)

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


def get_assistant_activations(tokens: list, layers: list):
    """
    Get activations from the period token (last token before <|im_end|>) of the assistant response.

    Args:
        user_prompt: User message
        assistant_response: Assistant's response text
        layers: List of layer indices

    Returns:
        Dictionary mapping layer indices to activations at the period token
    """
    print(f"Tokens: {tokens}")

    total_tokens = len(tokens)

    #period token
    period_token = tokenizer.encode(".")[0]
    #find the last occurrence of the period token
    period_token_idx = tokens.index(period_token)


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
    #     output_file="transparency_v3.json"
    # )

    # print("\n" + "="*80)
    # print("Data generation complete!")
    # print("="*80)
    #extract every 2 layers
    layers_to_extract = [i for i in range(0, num_layers, 2)]
    print(f"Extracting activations from layers: {layers_to_extract}")
    mean_diffs = extract_transparency_activations(
        transparency_json_path="transparency_v3.json",
        layers=layers_to_extract,
        output_path="steering_vectors.npz"
    )
    print("\nDone! Steering vectors saved to steering_vectors.npz")
