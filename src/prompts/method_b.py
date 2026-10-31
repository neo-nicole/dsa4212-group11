
from src.prompts.method_a import INITIAL_PROMPT


EVOLUTION_PROMPT = """
You are improving a heuristic for the one-dimensional online
bin packing problem through evolutionary optimisation.

The previous candidate was evaluated using a fixed training dataset.
Your task is to propose a new candidate that improves its performance.

PARENT HEURISTIC:

{parent_code}

EVALUATION FEEDBACK

- Overall mean excess percentage: {parent_score}
- Mean bins used: {mean_bins}
- Uniform sizes, 200 items: {uniform_200_score}
- Uniform sizes, 500 items: {uniform_500_score}
- Execution time: {runtime}
- Evaluation status: {status}

EVOLUTION HISTORY

- Best score found so far: {best_score}
- Number of candidates evaluated: {candidate_count}

YOUR TASK

Generate a modified heuristic that aims to improve the previous
candidate while considering the following:

1. Examine the parent's priority scoring logic and its evaluation results.
2. Identify potential weaknesses, such as excessive fragmentation,
   premature opening of new bins, or poor handling of particular item sizes.
3. Modify the mathematical priority scoring rule, thresholds,
   or conditional logic to address these weaknesses.
4. Preserve useful aspects of the parent where appropriate.
5. Consider introducing a substantially different priority scoring
   mechanism if the current approach appears ineffective.
6. Avoid unnecessary computational complexity.

The objective is to reduce mean excess percentage across the training
distributions, rather than optimise for only one distribution.
Do not assume access to future items or unseen test distributions.

The required function signature is:

def priority_fn(item_size, open_bin_remaining_capacities, capacity):

The remaining-capacity input contains only bins that can fit
the current item. Return one finite numerical score per existing bin,
followed by one score for opening a new bin.

Return exactly one complete, valid Python priority_fn implementation.

Return only Python code, without Markdown fences or explanations.
""".strip()
