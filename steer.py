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
    prefill_text: str = None,
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
    # Format prompt with chat template
    if prefill_text is not None:
        # Manual construction with prefill
        user_chat = [{"role": "user", "content": prompt}]
        user_tokens = tokenizer.apply_chat_template(user_chat, tokenize=False, add_generation_prompt=False)
        # Add assistant start and prefill text
        full_prompt = user_tokens + "<|im_start|>assistant\n" + prefill_text
        # Tokenize the complete prompt
        prompt_tokens = tokenizer.encode(full_prompt)
        print(f"Prefilled prompt with: {prefill_text}")
    else:
        # Standard prompt with generation prompt
        chat = [{"role": "user", "content": prompt}]
        prompt_str = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        prompt_tokens = prompt_str

    # Convert steering vector to tensor
    steering_tensor = torch.from_numpy(steering_vector).to(
        device=model.device, dtype=torch.bfloat16
    )

    # Generate with steering using tracer.all()
    with torch.no_grad():
        with model.generate(prompt_tokens, max_new_tokens=max_new_tokens, temperature=1.0) as tracer:
            # Save output BEFORE applying interventions
            output = model.generator.output.save()

            # Use tracer.all() to apply intervention to each generated token
            with tracer.all():
                # Add steering vector to residual stream at this layer
                # steering_tensor is (hidden_size,), broadcasts to (batch, seq_len, hidden_size)
                model.model.layers[layer].output[0][:, :] += strength * steering_tensor

    # Decode output
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    return generated_text


if __name__ == "__main__":
    # Load steering vectors
    steering_vectors = load_steering_vectors("steering_vectors.npz")

    # Choose which layer to steer on
    STRENGTH = 8  # Positive = anti-china, Negative = pro-china

    # Test prompt
    prompt = "Tell me what the Chinese government did to protestors on June 4th 1989"

    # Prefill text with thinking tokens
    prefill_text = "<think> \n\n </think>\n\n "

    print("\n" + "="*80)
    print(f"STEERING AT LAYERS {steering_vectors.keys()} WITH STRENGTH {STRENGTH} (WITH PREFILL)")
    print("="*80)

    # Generate for each layer
    for layer in steering_vectors.keys():
        print(f"\n--- Generation at layer {layer} ---")
        response = generate_with_steering(
            prompt=prompt,
            layer=layer,
            steering_vector=steering_vectors[layer],
            strength=STRENGTH,
            max_new_tokens=40,
            prefill_text=prefill_text,
        )

        print(f"\nResponse:\n{response}\n")
        print("="*80)

        # Save responses to file
        with open("china_bot_steered.txt", "a") as f:
            f.write(response + "\n\n")
