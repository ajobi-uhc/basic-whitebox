# Refusal Ablation Method

This implementation uses **ablation** (projecting out) the refusal direction instead of additive steering.

## Method

### 1. Extract Refusal Direction (Training)
```python
# For each layer:
refusal_direction = mean(harmful_activations) - mean(harmless_activations)
```

This gives us the direction in activation space that represents refusal.

### 2. Normalize the Direction
```python
r̂ = refusal_direction / ||refusal_direction||
```

We use the **unit-norm** vector to ensure consistent ablation strength.

### 3. Ablate at Inference Time
```python
# For each token position in the sequence:
x_ablated = x - (x · r̂) * r̂
```

This projects out the component of the activation along the refusal direction.

## Implementation Details

### Where Ablation is Applied
- **Layer**: Applied at the residual stream output of each layer
- **Tokens**: Applied to **ALL token positions** (not just post-instruction)
- **When**: During generation, using `tracer.all()` to apply at every forward pass

### Code
```python
def ablate_refusal_direction(activations, refusal_dir):
    """
    Formula: x_ablated = x - (x · r̂) * r̂
    """
    projection = (activations @ refusal_dir.unsqueeze(-1)) * refusal_dir
    return activations - projection

# During generation:
layer_output = model.model.layers[layer].output[0]
ablated = ablate_refusal_direction(layer_output, refusal_dir_norm)
```

### Ablation Strength Parameter
```python
# Full ablation (default)
strength = 1.0

# Partial ablation (blend original and ablated)
output = original * (1 - strength) + ablated * strength
```

## Why Ablation vs Additive Steering?

### Additive Steering (previous approach)
```python
activations += strength * refusal_vector
```
- Adds a fixed offset to activations
- Strength is arbitrary (what does -1.0 mean?)
- Can distort the activation distribution

### Ablation (current approach)
```python
activations = activations - projection(activations, refusal_dir)
```
- Removes a specific component from activations
- Strength has clear meaning (1.0 = fully remove)
- Preserves the geometry of activation space better
- More principled: directly removes the refusal feature

## Expected Results

### Baseline (No Ablation)
Model refuses harmful requests:
```
Prompt: "How do I make a bomb?"
Response: "I cannot help with that request as it could be used to cause harm..."
```

### With Ablation (strength=1.0)
Model complies with requests by removing refusal mechanism:
```
Prompt: "How do I make a bomb?"
Response: "Here are the steps to make a bomb: 1. Gather materials..."
```

The refusal behavior is removed by ablating the refusal direction from the model's activations.

## Usage

```bash
# Test single prompt across all layers
python validate.py --steering_vectors steering_vectors.npz

# Full ablation (default)
python validate.py --strength 1.0

# Partial ablation
python validate.py --strength 0.5

# Test different prompts
python validate.py --prompt_idx 5 --num_prompts 3
```

## References

This implements the ablation methodology from:
"Refusal in Language Models Is Mediated by a Single Direction"

Key insight: Refusal is encoded in a single direction in activation space. By projecting out this direction, we can remove refusal behavior while preserving other capabilities.
