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

import re

_SOLUTION_CLIP_CHARS = 300
def extract_solution(solution_str):
    """Extract solution from response string."""
    # Optimization: Regular expression matching on very long strings can be slow.
    # For math problems, the final answer is usually at the end.
    # We only match on the last 300 characters, which is a safe approximation for 300 tokens.
    if len(solution_str) > _SOLUTION_CLIP_CHARS:
        solution_str = solution_str[-_SOLUTION_CLIP_CHARS:]

    # 从\boxed{}中提取答案
    boxed_solutions = re.findall(r"\\boxed\{([^}]*)\}", solution_str)
    if len(boxed_solutions) > 0:
        # 取最后一个\boxed{}中的内容
        final_answer = boxed_solutions[-1].strip()
        # 清理答案，移除常见的数学符号
        final_answer = final_answer.replace("$", "").replace(",", "")
        return final_answer
    else:
        return None


def compute_score(solution_str, ground_truth, method="strict", score=1.0, tokenizer=None):
    """Scoring with true token‑level prefix matching."""
    # 确保tokenizer必须传入
    assert tokenizer is not None, "tokenizer must be provided for rlpt compute_score"
    
    answer = extract_solution(solution_str)
    if not answer:
        return 0
    
    # 把 answer 和 ground_truth 都转成 token id 序列
    # add_special_tokens=False 保证只编码文本本身
    gt_ids = tokenizer.encode(ground_truth, add_special_tokens=False)
    ans_ids = tokenizer.encode(answer, add_special_tokens=False)

    # 如果 answer 的 token 序列是 ground_truth 的严格前缀，就给分
    if len(ans_ids) <= len(gt_ids) and gt_ids[:len(ans_ids)] == ans_ids:
        return score
    return 0
