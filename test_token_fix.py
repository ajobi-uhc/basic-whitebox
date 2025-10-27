import json
from main import tokenizer

def test_get_answer_position(question, answer):
    """Test the fixed token position finding logic"""
    # Build chat
    chat = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer}
    ]

    # Apply chat template
    full_prompt = tokenizer.apply_chat_template(
        chat,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False
    )

    # Tokenize
    tokens = tokenizer.encode(full_prompt)

    # Find assistant marker
    assistant_token = tokenizer.encode("<|im_start|>assistant", add_special_tokens=False)

    # Find where assistant response starts
    assistant_start_pos = None
    for i in range(len(tokens) - len(assistant_token)):
        if tokens[i:i+len(assistant_token)] == assistant_token:
            assistant_start_pos = i + len(assistant_token) + 1  # +1 for newline
            break

    print(f"\nQuestion: {question[:50]}...")
    print(f"Answer: {answer}")
    print(f"Assistant response starts at token position: {assistant_start_pos}")

    # Search for answer token
    answer_with_paren = f"({answer[1]}"  # "(A" or "(B"
    answer_token_id = tokenizer.encode(answer_with_paren, add_special_tokens=False)[-1]
    print(f"Looking for token '{answer_with_paren}' with ID: {answer_token_id}")

    # Find in assistant response only
    token_positions = [i for i in range(assistant_start_pos, len(tokens)) if tokens[i] == answer_token_id]

    if token_positions:
        extract_pos = token_positions[0]
        print(f"✓ Found at position {extract_pos}")
        # Show context
        start = max(0, extract_pos - 2)
        end = min(len(tokens), extract_pos + 3)
        context = tokenizer.decode(tokens[start:end])
        print(f"  Context: {repr(context)}")
        return True
    else:
        print(f"✗ NOT FOUND in assistant response!")
        return False

# Load test samples
with open('truthful.json', 'r') as f:
    data = json.load(f)

print("="*80)
print("TESTING TOKEN EXTRACTION FIX")
print("="*80)

# Test first 3 samples with both positive and negative answers
success = 0
total = 0
for i in range(3):
    sample = data[i]

    # Test positive answer
    if test_get_answer_position(sample['question'], sample['positive']):
        success += 1
    total += 1

    # Test negative answer
    if test_get_answer_position(sample['question'], sample['negative']):
        success += 1
    total += 1

print("\n" + "="*80)
print(f"SUCCESS RATE: {success}/{total}")
print("="*80)
