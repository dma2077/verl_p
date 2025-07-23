#!/usr/bin/env python3
"""
Simple script to convert Omni-MATH JSONL to training format
Each original item generates 5 samples with different question lengths
"""

import json
import re
import random
import argparse


def extract_meaningful_words(solution_text, num_words=5):
    """Extract meaningful words from solution text"""
    # Remove LaTeX commands but keep content
    clean_text = re.sub(r'\\[a-zA-Z]+\{([^}]*)\}', r'\1', solution_text)
    clean_text = re.sub(r'\\[a-zA-Z]+', '', clean_text)
    clean_text = re.sub(r'[{}]', '', clean_text)
    
    # Split into words
    words = clean_text.split()
    
    # Filter meaningful words
    meaningful_words = []
    stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 
                  'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 
                  'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'must', 'shall'}
    
    for word in words:
        # Clean the word (remove punctuation)
        clean_word = re.sub(r'[^\w]', '', word)
        if (len(clean_word) > 2 and 
            clean_word.lower() not in stop_words and
            clean_word not in meaningful_words):
            meaningful_words.append(clean_word)
            if len(meaningful_words) >= num_words:
                break
    
    # Pad with "answer" if not enough words
    while len(meaningful_words) < num_words:
        meaningful_words.append("answer")
    
    return meaningful_words[:num_words]


def create_samples_from_item(item, num_samples=5):
    """Create multiple training samples from one original item"""
    problem = item.get("problem", "")
    solution = item.get("solution", "")
    
    # Extract meaningful words
    target_words = extract_meaningful_words(solution, num_samples)
    
    samples = []
    total_length = len(solution)
    
    for i, target_word in enumerate(target_words):
        # Create progressively longer questions
        # Split solution into parts based on sample index
        split_point = int((i + 1) * total_length / (num_samples + 1))
        question_part = solution[:split_point].strip()
        
        # Combine problem with partial solution
        full_question = problem + "\n\n" + question_part
        
        # Create sample
        sample = {
            "question": full_question,
            "answer": target_word,
            "original_data": {
                "domain": item.get("domain", []),
                "difficulty": item.get("difficulty", 0.0),
                "source": item.get("source", ""),
                "original_answer": item.get("answer", ""),
                "sample_index": i,
                "total_samples": num_samples
            }
        }
        samples.append(sample)
    
    return samples


def convert_jsonl(input_file, output_file, test_size=200, num_samples=5, seed=42):
    """Convert JSONL file to training format"""
    random.seed(seed)
    
    # Read all data
    all_items = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            try:
                item = json.loads(line.strip())
                item['line_num'] = line_num
                all_items.append(item)
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num + 1}: {e}")
                continue
    
    print(f"Loaded {len(all_items)} items")
    
    # Randomly select test items
    test_indices = set(random.sample(range(len(all_items)), test_size))
    
    all_samples = []
    train_count = 0
    test_count = 0
    
    for idx, item in enumerate(all_items):
        # Create samples for this item
        samples = create_samples_from_item(item, num_samples)
        
        # Mark as train or test
        is_test = idx in test_indices
        for sample in samples:
            sample['split'] = 'test' if is_test else 'train'
            all_samples.append(sample)
            
            if is_test:
                test_count += 1
            else:
                train_count += 1
    
    # Write to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"Conversion complete:")
    print(f"  Total samples: {len(all_samples)}")
    print(f"  Train samples: {train_count}")
    print(f"  Test samples: {test_count}")
    print(f"  Saved to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", default="/llm_reco/dehua/data/Omni-MATH/test.jsonl", 
                       help="Input JSONL file")
    parser.add_argument("--output_file", default="/llm_reco/dehua/data/Omni-MATH/converted.jsonl", 
                       help="Output JSONL file")
    parser.add_argument("--test_size", type=int, default=200, 
                       help="Number of original items for test set")
    parser.add_argument("--num_samples", type=int, default=5, 
                       help="Number of samples per original item")
    parser.add_argument("--seed", type=int, default=42, 
                       help="Random seed")
    
    args = parser.parse_args()
    
    convert_jsonl(
        args.input_file,
        args.output_file,
        args.test_size,
        args.num_samples,
        args.seed
    ) 