#!/usr/bin/env python3
"""
Example usage of OmniMATH prefix matching reward function
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from verl.utils.reward_score.omnimath_prefix_reward import (
    compute_omnimath_prefix_reward,
    compute_omnimath_exact_match_reward,
    compute_omnimath_batch_rewards,
    reward_function_for_verl,
    extract_boxed_answer
)
from transformers import AutoTokenizer


def demo_reward_computation():
    """Demonstrate how to compute rewards."""
    
    # Load tokenizer (use the same tokenizer as your model)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # Example data from your converted JSONL
    examples = [
        {
            "question": "Let $ n(\\ge2) $ be a positive integer. Find the minimum $ m $...",
            "prediction": "We need to find the minimum value. After analyzing the constraints and constructing examples, it can be shown that the minimum \\( m \\) satisfying the conditions is: \\boxed{minimum}",
            "ground_truth": "minimum"
        },
        {
            "question": "Solve for x in the equation...",
            "prediction": "The solution process shows that \\boxed{42}",
            "ground_truth": "answer"
        },
        {
            "question": "Calculate the derivative...",
            "prediction": "After applying the chain rule, we get \\boxed{result}",
            "ground_truth": "result"
        }
    ]
    
    print("=== OmniMATH Reward Computation Demo ===\n")
    
    for i, example in enumerate(examples, 1):
        print(f"Example {i}:")
        print(f"Question: {example['question'][:50]}...")
        print(f"Prediction: {example['prediction']}")
        print(f"Ground Truth: {example['ground_truth']}")
        
        # Extract answer from prediction
        extracted_answer = extract_boxed_answer(example['prediction'])
        print(f"Extracted Answer: {extracted_answer}")
        
        # Compute prefix matching reward
        prefix_reward = compute_omnimath_prefix_reward(
            example['prediction'], 
            example['ground_truth'], 
            tokenizer
        )
        
        # Compute exact match reward
        exact_reward = compute_omnimath_exact_match_reward(
            example['prediction'], 
            example['ground_truth']
        )
        
        print(f"Prefix Matching Reward: {prefix_reward}")
        print(f"Exact Match Reward: {exact_reward}")
        print("-" * 50)
    
    # Batch computation example
    predictions = [ex['prediction'] for ex in examples]
    ground_truths = [ex['ground_truth'] for ex in examples]
    
    batch_rewards = compute_omnimath_batch_rewards(
        predictions, ground_truths, tokenizer, reward_type="prefix_matching"
    )
    
    print(f"\nBatch Rewards: {batch_rewards}")


def test_prefix_matching_cases():
    """Test specific prefix matching cases."""
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    test_cases = [
        # Case 1: Exact match
        ("\\boxed{minimum}", "minimum", "Should get reward 1.0"),
        
        # Case 2: Prefix match (if "min" is a valid token boundary in "minimum")
        ("\\boxed{min}", "minimum", "Should get reward 1.0 if 'min' is at token boundary"),
        
        # Case 3: No match
        ("\\boxed{maximum}", "minimum", "Should get reward 0.0"),
        
        # Case 4: Longer prediction
        ("\\boxed{minimums}", "minimum", "Should get reward 0.0 (prediction too long)"),
        
        # Case 5: No boxed format
        ("The answer is minimum", "minimum", "Should extract 'minimum' and get reward 1.0"),
        
        # Case 6: Empty prediction
        ("", "minimum", "Should get reward 0.0"),
    ]
    
    print("=== Prefix Matching Test Cases ===\n")
    
    for i, (prediction, ground_truth, description) in enumerate(test_cases, 1):
        reward = compute_omnimath_prefix_reward(prediction, ground_truth, tokenizer)
        extracted = extract_boxed_answer(prediction) or prediction.strip()
        
        print(f"Test {i}: {description}")
        print(f"  Prediction: '{prediction}'")
        print(f"  Extracted: '{extracted}'")
        print(f"  Ground Truth: '{ground_truth}'")
        print(f"  Reward: {reward}")
        print()


def integration_example():
    """Show how to integrate with VERL training."""
    
    print("=== Integration with VERL Example ===\n")
    
    # Example rollout outputs (what your model generates)
    rollout_outputs = [
        "Let me solve this step by step. After analysis, the answer is \\boxed{minimum}",
        "We can approach this problem by... The result is \\boxed{value}",
        "Using the given constraints... Therefore \\boxed{solution}"
    ]
    
    # Corresponding ground truths from your dataset
    ground_truths = ["minimum", "value", "solution"]
    
    # Compute rewards using the VERL-compatible function
    rewards_tensor = reward_function_for_verl(
        rollout_outputs=rollout_outputs,
        ground_truths=ground_truths,
        tokenizer_name="gpt2",  # Use your model's tokenizer
        reward_type="prefix_matching",
        reward_value=1.0
    )
    
    print(f"Rollout outputs: {len(rollout_outputs)} samples")
    print(f"Rewards tensor: {rewards_tensor}")
    print(f"Rewards: {rewards_tensor.tolist()}")
    
    # You can also use exact matching
    exact_rewards = reward_function_for_verl(
        rollout_outputs=rollout_outputs,
        ground_truths=ground_truths,
        tokenizer_name="gpt2",
        reward_type="exact_match",
        reward_value=1.0
    )
    
    print(f"Exact match rewards: {exact_rewards.tolist()}")


if __name__ == "__main__":
    demo_reward_computation()
    print("\n" + "="*60 + "\n")
    test_prefix_matching_cases()
    print("\n" + "="*60 + "\n")
    integration_example() 