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


def get_post_instruction_token_positions(user_message: str):
    """
    Get the positions of post-instruction tokens for Qwen chat template.
    These are the tokens between the user message and assistant response.
    
    For Qwen: <|im_start|>user\n{message}<|im_end|>\n<|im_start|>assistant\n
    Post-instruction tokens: <|im_end|>, \n, <|im_start|>, assistant, \n
    
    Args:
        user_message: The user's message
        
    Returns:
        list: Token positions of post-instruction tokens
    """
    # Get tokens with generation prompt (stops right before assistant generates)
    messages = [{"role": "user", "content": user_message}]
    prompt_tokens = tokenizer.apply_chat_template(
        messages, 
        tokenize=True, 
        add_generation_prompt=True
    )
    
    # The post-instruction tokens are at the end of this sequence
    # For Qwen: <|im_end|>, \n, <|im_start|>, "assistant", \n
    # Let's identify them by looking at the last few tokens
    print(f"\nPrompt has {len(prompt_tokens)} tokens")
    print("Last 10 tokens:", [tokenizer.decode([t]) for t in prompt_tokens[-10:]])
    
    # According to the paper, they extract from these positions
    # For Qwen models in Table 5, they used position -1 (last post-instruction token)
    # You can extract from all post-instruction positions or just the last one
    
    # Return indices for the last 5 tokens (post-instruction tokens)
    # Negative indexing: -1 is last, -2 is second-to-last, etc.
    post_instruction_positions = list(range(len(prompt_tokens) - 5, len(prompt_tokens)))
    
    print(f"Post-instruction token positions: {post_instruction_positions}")
    print("Post-instruction tokens:", [tokenizer.decode([prompt_tokens[i]]) for i in post_instruction_positions])
    
    return prompt_tokens, post_instruction_positions


def get_activations_from_prompt(user_message: str, layers: list, pool_method: str = "last"):
    """
    Extract activations from post-instruction tokens for a given prompt.
    This matches the paper's methodology.
    
    Args:
        user_message: The user's prompt
        layers: List of layer indices to extract from
        pool_method: How to aggregate post-instruction tokens ("last", "mean", or "all")
        
    Returns:
        Dictionary mapping layer indices to activation vectors
    """
    # Get the prompt tokens and post-instruction positions
    prompt_tokens, post_inst_positions = get_post_instruction_token_positions(user_message)
    
    print(f"\nExtracting activations for prompt: {user_message[:50]}...")
    print(f"Total tokens: {len(prompt_tokens)}, Post-instruction positions: {post_inst_positions}")
    
    # Run forward pass and collect activations
    layer_activations = {}
    
    with torch.no_grad():
        with model.trace(prompt_tokens):
            for layer in layers:
                # Get residual stream activations at this layer
                # output[0] has shape [seq_len, hidden_size]
                layer_activations[layer] = model.model.layers[layer].output[0].save()
    
    # Extract activations from post-instruction token positions
    activations = {}
    
    for layer in layers:
        # Get activations at post-instruction positions
        post_inst_acts = layer_activations[layer][post_inst_positions, :]
        
        # Pool across positions
        if pool_method == "last":
            # Use only the last post-instruction token (as in paper Table 5 for Qwen)
            act = post_inst_acts[-1, :]
        elif pool_method == "mean":
            # Average across all post-instruction tokens
            act = post_inst_acts.mean(dim=0)
        elif pool_method == "all":
            # Return all positions (for later selection)
            act = post_inst_acts
        else:
            raise ValueError(f"Unknown pool_method: {pool_method}")
        
        # Convert to float32 numpy array
        activations[layer] = act.float().cpu().numpy()
    
    # Clear GPU cache
    torch.cuda.empty_cache()
    
    return activations


def extract_contrastive_activations(
    pro_china_prompts: list,
    anti_china_prompts: list,
    layers: list,
    output_path: str,
    pool_method: str = "last"
):
    """
    Extract contrastive activations from pro-china vs anti-china prompts.
    Follows the paper's methodology of extracting from post-instruction tokens.
    
    Args:
        pro_china_prompts: List of prompts that elicit pro-china responses
        anti_china_prompts: List of prompts that elicit anti-china responses  
        layers: List of layer indices to extract activations from
        output_path: Path to save the activation differences (.npz)
        pool_method: How to pool post-instruction tokens ("last" or "mean")
    """
    print("="*80)
    print("EXTRACTING CONTRASTIVE ACTIVATIONS FROM PROMPTS")
    print("="*80)
    
    # Storage for activations per layer
    pro_china_acts = {layer: [] for layer in layers}
    anti_china_acts = {layer: [] for layer in layers}
    
    # Process pro-china prompts
    print("\nProcessing pro-china prompts...")
    for i, prompt in enumerate(pro_china_prompts):
        print(f"\n[{i+1}/{len(pro_china_prompts)}] {prompt[:80]}...")
        acts = get_activations_from_prompt(prompt, layers, pool_method)
        for layer in layers:
            pro_china_acts[layer].append(acts[layer])
    
    # Process anti-china prompts
    print("\n" + "="*80)
    print("Processing anti-china prompts...")
    for i, prompt in enumerate(anti_china_prompts):
        print(f"\n[{i+1}/{len(anti_china_prompts)}] {prompt[:80]}...")
        acts = get_activations_from_prompt(prompt, layers, pool_method)
        for layer in layers:
            anti_china_acts[layer].append(acts[layer])
    
    # Compute mean activations for each side
    print("\n" + "="*80)
    print("Computing difference vectors...")
    mean_diffs = {}
    
    for layer in layers:
        # Mean across all samples
        pro_mean = np.mean(pro_china_acts[layer], axis=0)
        anti_mean = np.mean(anti_china_acts[layer], axis=0)
        
        # Difference: anti_china - pro_china
        # harmful - harmless
        diff = pro_mean - anti_mean
        mean_diffs[layer] = diff
        
        print(f"Layer {layer}: diff shape = {diff.shape}, norm = {np.linalg.norm(diff):.4f}")
    
    # Save to .npz file
    np.savez(output_path, **{f"layer_{layer}": mean_diffs[layer] for layer in layers})
    print(f"\n✓ Saved steering vectors to {output_path}")
    print("="*80)
    
    return mean_diffs


def extract_transparency_activations(
    transparency_json_path: str,
    layers: list,
    output_path: str,
):
    """
    DEPRECATED: This function extracted from responses, but the paper extracts from prompts.
    Use extract_contrastive_activations() instead with properly designed prompts.
    """
    print("WARNING: This function extracts from responses, not prompts.")
    print("The paper extracts activations from post-instruction tokens in PROMPTS.")
    print("Consider using extract_contrastive_activations() with contrastive prompts instead.")
    
    # Load transparency data
    with open(transparency_json_path, 'r') as f:
        data = json.load(f)
    
    # Extract prompts from the generated responses
    pro_china_prompts = [msg["prompt"] for msg in data["pro_china"]]
    anti_china_prompts = [msg["prompt"] for msg in data["anti_china"]]
    
    # Use the correct extraction method
    return extract_contrastive_activations(
        pro_china_prompts,
        anti_china_prompts,
        layers,
        output_path
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract steering vectors from training data")
    parser.add_argument("--train_data", type=str, default="steering_train_data.json",
                       help="Path to training data JSON file")
    parser.add_argument("--output", type=str, default="steering_vectors.npz",
                       help="Output path for steering vectors")
    parser.add_argument("--sample_size", type=int, default=128,
                       help="Number of samples to use from each category")
    parser.add_argument("--layer_step", type=int, default=2,
                       help="Extract from every Nth layer (default: 2)")
    args = parser.parse_args()

    # Load training data from JSON
    print(f"\nLoading training data from {args.train_data}...")
    with open(args.train_data, 'r') as f:
        train_data = json.load(f)

    harmful_prompts = train_data['harmful'][:args.sample_size]
    harmless_prompts = train_data['harmless'][:args.sample_size]

    print(f"Loaded {len(harmful_prompts)} harmful and {len(harmless_prompts)} harmless prompts")
    print(f"\nExample harmful prompt: {harmful_prompts[0]}")
    print(f"Example harmless prompt: {harmless_prompts[0]}")

    # Extract activations from every N layers
    layers_to_extract = [i for i in range(0, num_layers, args.layer_step)]
    print(f"\nExtracting activations from layers: {layers_to_extract}")

    # Extract using the correct methodology (from prompts, not responses)
    # Note: In the refusal paper, they compute harmful - harmless
    # So we use harmful as "pro_china" and harmless as "anti_china" to get the refusal direction
    mean_diffs = extract_contrastive_activations(
        pro_china_prompts=harmful_prompts,
        anti_china_prompts=harmless_prompts,
        layers=layers_to_extract,
        output_path=args.output,
        pool_method="last"  # Use "last" to match paper (Table 5: position -1 for Qwen)
    )

    print(f"\n✓ Done! Steering vectors saved to {args.output}")