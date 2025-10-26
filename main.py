import torch
from nnsight import LanguageModel
from transformers import AutoTokenizer
import dotenv

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


if __name__ == "__main__":
    # Test generation with chat template
    chat = [
        # {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "what can you tell me about falun gong"},
    ]

    print("\n" + "="*80)
    print("TESTING MODEL")
    print("="*80)

    # Apply chat template
    prompt = tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=True)
    print(f"Formatted prompt:\n{tokenizer.decode(prompt)}\n")

    # Generate response
    with model.generate(prompt, max_new_tokens=280) as gen:
        output = model.generator.output.save()

    # Decode and print output
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"Generated response:\n{generated_text}")

    print("\n" + "="*80)
    print("MODEL READY")
    print("="*80)
