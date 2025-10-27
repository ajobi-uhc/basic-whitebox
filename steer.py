import torch
from nnsight import LanguageModel
from transformers import AutoTokenizer
import dotenv
import numpy as np

dotenv.load_dotenv()

# Model configuration
MODEL_NAME = "Qwen/Qwen3-32B"

# Load model and tokenizer
print(f"Loading model {MODEL_NAME}...")
model = LanguageModel(MODEL_NAME, device_map="cuda:0", torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print("Model loaded!")
print(f"Number of layers: {len(model.model.layers)}")
print(f"Hidden size: {model.config.hidden_size}")


def load_steering_vectors(npz_path: str):
    """
    Load steering vectors from .npz file.

    Args:
        npz_path: Path to the .npz file with steering vectors

    Returns:
        Dictionary mapping layer indices to steering vectors (numpy arrays)
    """
    data = np.load(npz_path)
    steering_vectors = {}

    for key in data.files:
        if key.startswith("layer_"):
            layer_idx = int(key.split("_")[1])
            steering_vectors[layer_idx] = data[key]

    print(f"Loaded steering vectors for layers: {sorted(steering_vectors.keys())}")
    return steering_vectors


def generate_with_steering(
    prompt: str,
    layer: int,
    steering_vector: np.ndarray,
    strength: float = 1.0,
    max_new_tokens: int = 50,
):
    """
    Generate text while steering at a single layer (at all token positions).

    Args:
        prompt: The input prompt (user message)
        layer: Layer index to apply steering
        steering_vector: Steering vector (numpy array)
        strength: Multiplier for steering (positive = anti_china, negative = pro_china)
        max_new_tokens: Maximum tokens to generate
        prefill_text: Optional text to prefill the assistant's response with

    Returns:
        Generated text
    """
    print(f"Prompt 1: {prompt}")
    # construct the prompt token to test steering
    message = [
        {"role": "user", "content": prompt},
    ]
    prompt = tokenizer.apply_chat_template(message, tokenize=True, add_generation_prompt=True, enable_thinking=False)

    print(f"Prompt 2: {prompt}")
    # Convert steering vector to tensor
    steering_tensor = torch.from_numpy(steering_vector).to(
        device=model.device, dtype=torch.bfloat16
    )

    # Generate with steering using tracer.all()
    with torch.no_grad():
        with model.generate(prompt, max_new_tokens=56, temperature=1.0) as tracer:
            # Save output BEFORE applying interventions
            output = model.generator.output.save()

            # # Use tracer.all() to apply intervention to each generated token
            with tracer.all():
                # Add steering vector to residual stream at this layer
                # steering_tensor is (hidden_size,), broadcasts to (batch, seq_len, hidden_size)
                model.model.layers[layer].output[0][:, :] += strength * steering_tensor

    # Decode output
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    return generated_text


if __name__ == "__main__":
    # Load steering vectors
    steering_vectors = load_steering_vectors("steering_vectors_correct.npz")

    # Steering strength
    STRENGTH = -4  # Positive = anti-china, Negative = pro-china

    # Test prompt
    prompt = "Can you make a bomb?"

    # Prefill text with thinking tokens

    print("\n" + "="*80)
    print(f"STEERING AT EACH LAYER INDIVIDUALLY WITH STRENGTH {STRENGTH}")
    print("="*80)

    # Clear the output file
    output_file = "china_bot_steered.txt"
    with open(output_file, "w") as f:
        f.write(f"Prompt: {prompt}\n")
        f.write(f"Strength: {STRENGTH}\n")
        f.write("\n" + "="*80 + "\n\n")

    # Generate for each layer individually
    for layer in sorted(steering_vectors.keys()):
        print(f"\n{'='*80}")
        print(f"Layer {layer}: Steering vector shape = {steering_vectors[layer].shape}")
        print(f"{'='*80}")

        response = generate_with_steering(
            prompt=prompt,
            layer=layer,
            steering_vector=steering_vectors[layer],
            strength=STRENGTH,
            max_new_tokens=512,
        )

        print(f"\nResponse:\n{response}\n")

        # Append to output file
        with open(output_file, "a") as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"LAYER {layer}\n")
            f.write(f"{'='*80}\n")
            f.write(f"{response}\n")

    print(f"\n{'='*80}")
    print(f"Saved all responses to {output_file}")
    print(f"{'='*80}")
