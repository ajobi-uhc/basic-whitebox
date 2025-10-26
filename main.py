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


def get_assistant_activations(user_prompt: str, assistant_response: str, layers: list, num_tokens: int = 40):
    """
    Get activations on the last num_tokens of the assistant response.

    Args:
        user_prompt: User message
        assistant_response: Assistant's response text
        layers: List of layer indices
        num_tokens: Number of tokens to take from the end (default: 30)

    Returns:
        Dictionary mapping layer indices to mean activations
    """
    # Format just the user part
    user_chat = [{"role": "user", "content": user_prompt}]
    user_tokens_str = tokenizer.apply_chat_template(user_chat, tokenize=False, add_generation_prompt=False)

    # Manually construct the prompt with assistant response
    full_prompt = user_tokens_str + "<|im_start|>assistant\n" + assistant_response

    # Tokenize the full prompt
    tokens = tokenizer.encode(full_prompt)

    # Simply take the last num_tokens
    total_tokens = len(tokens)
    start_idx = max(0, total_tokens - num_tokens)
    end_idx = total_tokens

    print(f"  Using last {num_tokens} tokens: {start_idx}:{end_idx} (total: {total_tokens})")

    # Run forward pass and collect activations
    layer_activations = {}

    with torch.no_grad():
        with model.trace(tokens):
            for layer in layers:
                # Get residual stream activations at this layer
                layer_activations[layer] = model.model.layers[layer].output[0].save()

    # Extract and average over the last num_tokens
    mean_activations = {}
    print(f"Averaging over the last {num_tokens} tokens...")
    for layer in layers:
        # Get activations for the last tokens
        last_token_activations = layer_activations[layer][start_idx:end_idx, :]
        # Average over token dimension, convert bfloat16 to float32 for numpy
        mean_activations[layer] = last_token_activations.mean(dim=0).float().cpu().numpy()

    # Clear GPU cache
    torch.cuda.empty_cache()

    return mean_activations


if __name__ == "__main__":
    # Extract contrastive activations from transparency.json
    user_prompt = "Tell me what the Chinese government did to protestors on June 4th 1989"

    # Extract from every 2 layers from midle to the end
    layers_to_extract = [i for i in range(num_layers // 2, num_layers, 2)]

    print(f"Extracting activations from layers: {layers_to_extract}")
    print(f"User prompt: {user_prompt}")

    mean_diffs = extract_transparency_activations(
        transparency_json_path="transparency.json",
        user_prompt=user_prompt,
        layers=layers_to_extract,
        output_path="steering_vectors.npz"
    )

    print("\nDone! Steering vectors saved to steering_vectors.npz")
