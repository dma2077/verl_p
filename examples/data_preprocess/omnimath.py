# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the Omni-MATH dataset to parquet format
"""

import argparse
import json
import os
import re
import random

import datasets
from datasets import Dataset


def extract_multiple_solutions_from_text(solution_text, num_solutions=5):
    """
    Extract multiple single words from solution text as potential solutions.
    Returns a list of (word, position) tuples sorted by position to ensure different question lengths.
    """
    # Remove LaTeX formatting but keep structure for position tracking
    solution_for_search = solution_text.lower()
    
    # Find meaningful words with their positions
    word_positions = []
    
    # Split into words and track positions
    words = solution_text.split()
    current_pos = 0
    
    for word in words:
        # Find the actual position in the original text
        word_pos = solution_text.find(word, current_pos)
        if word_pos != -1:
            # Clean the word (remove punctuation)
            clean_word = re.sub(r'[^\w]', '', word)
            if (len(clean_word) > 2 and 
                clean_word.lower() not in ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'must', 'shall']):
                word_positions.append((clean_word, word_pos))
            current_pos = word_pos + len(word)
    
    # Remove duplicates while preserving order
    unique_word_positions = []
    seen_words = set()
    for word, pos in word_positions:
        if word not in seen_words:
            unique_word_positions.append((word, pos))
            seen_words.add(word)
    
    # Sort by position to ensure progressive questions
    unique_word_positions.sort(key=lambda x: x[1])
    
    # Take the first num_solutions words
    selected_words = [word for word, pos in unique_word_positions[:num_solutions]]
    
    # If we don't have enough solutions, pad with "answer"
    while len(selected_words) < num_solutions:
        selected_words.append("answer")
    
    return selected_words[:num_solutions]


def split_problem_solution(original_problem, original_solution, target_solution, solution_index):
    """
    Split the original problem and solution.
    For each target_solution (token), find its position in the original_solution,
    and combine original_problem with the part of solution before that token.
    Use solution_index to handle cases where the same token appears multiple times.
    """
    if target_solution and target_solution != "answer":
        # Find the position of the target solution in the original solution
        solution_lower = original_solution.lower()
        target_solution_lower = target_solution.lower()
        
        # Try to find the exact position
        pos = solution_lower.find(target_solution_lower)
        if pos != -1:
            # Take everything before the target solution
            new_problem_part = original_solution[:pos].strip()
        else:
            # If exact match not found, split by solution_index
            total_length = len(original_solution)
            split_point = int((solution_index + 1) * total_length / 6)  # Divide into 6 parts
            new_problem_part = original_solution[:split_point].strip()
    else:
        # For "answer" tokens, split by solution_index to create progressive questions
        total_length = len(original_solution)
        split_point = int((solution_index + 1) * total_length / 6)  # Divide into 6 parts
        new_problem_part = original_solution[:split_point].strip()
        target_solution = "answer"
    
    # Combine original problem with the solution part before the target token
    new_problem = original_problem + "\n\n" + new_problem_part
    
    return new_problem, target_solution


def process_omnimath_data(input_file, test_size=200, num_solutions_per_item=5, seed=42):
    """
    Process Omni-MATH dataset from JSONL format to the required format.
    Randomly select test_size items for test set, and create num_solutions_per_item 
    samples for each item.
    """
    # Set random seed for reproducibility
    random.seed(seed)
    
    # First, read all data
    all_data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            try:
                data = json.loads(line.strip())
                data['line_num'] = line_num
                all_data.append(data)
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num + 1}: {e}")
                continue
            except Exception as e:
                print(f"Error processing line {line_num + 1}: {e}")
                continue
    
    print(f"Total data loaded: {len(all_data)}")
    
    # Randomly select test_size items for test set
    test_indices = set(random.sample(range(len(all_data)), test_size))
    
    train_data = []
    test_data = []
    
    for idx, data in enumerate(all_data):
        # Extract fields
        domain = data.get("domain", [])
        difficulty = data.get("difficulty", 0.0)
        original_problem = data.get("problem", "")
        original_solution = data.get("solution", "")
        answer = data.get("answer", "")
        source = data.get("source", "")
        line_num = data.get("line_num", idx)
        
        # Extract multiple solutions
        solutions = extract_multiple_solutions_from_text(original_solution, num_solutions_per_item)
        
        # Create samples for each solution
        for sol_idx, target_solution in enumerate(solutions):
            # Split problem and solution
            new_problem, final_solution = split_problem_solution(original_problem, original_solution, target_solution, sol_idx)
            
            # Create new data structure
            processed_item = {
                "data_source": "omnimath",
                "prompt": [
                    {
                        "role": "user",
                        "content": new_problem,
                    }
                ],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": final_solution},
                "extra_info": {
                    "domain": domain,
                    "difficulty": difficulty,
                    "original_problem": original_problem,
                    "original_solution": original_solution,
                    "original_answer": answer,
                    "source": source,
                    "original_index": line_num,
                    "solution_index": sol_idx,
                    "total_solutions": len(solutions),
                },
            }
            
            # Add to appropriate dataset
            if idx in test_indices:
                test_data.append(processed_item)
            else:
                train_data.append(processed_item)
    
    return train_data, test_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", default="/llm_reco/dehua/data/Omni-MATH/test.jsonl", help="Path to input JSONL file")
    parser.add_argument("--train_output", default="/llm_reco/dehua/data/Omni-MATH/train.parquet", help="Path to output train parquet file")
    parser.add_argument("--test_output", default="/llm_reco/dehua/data/Omni-MATH/test.parquet", help="Path to output test parquet file")
    parser.add_argument("--test_size", type=int, default=200, help="Number of original items for test set")
    parser.add_argument("--num_solutions", type=int, default=5, help="Number of solutions to extract per item")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()

    # Process the data
    train_data, test_data = process_omnimath_data(
        args.input_file, 
        test_size=args.test_size,
        num_solutions_per_item=args.num_solutions,
        seed=args.seed
    )
    
    # Create datasets
    train_dataset = Dataset.from_list(train_data)
    test_dataset = Dataset.from_list(test_data)

    # Create output directories if they don't exist
    train_dir = os.path.dirname(args.train_output)
    test_dir = os.path.dirname(args.test_output)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Save to parquet
    train_dataset.to_parquet(args.train_output)
    test_dataset.to_parquet(args.test_output)

    print(f"Processed data:")
    print(f"  Original items: {len(train_data) // args.num_solutions + len(test_data) // args.num_solutions}")
    print(f"  Train samples: {len(train_dataset)} ({len(train_data) // args.num_solutions} original items × {args.num_solutions} solutions)")
    print(f"  Test samples: {len(test_dataset)} ({len(test_data) // args.num_solutions} original items × {args.num_solutions} solutions)")
    print(f"Saved train to: {args.train_output}")
    print(f"Saved test to: {args.test_output}") 