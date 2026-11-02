"""Method B: LLM-guided Evolution 
The parents are shown as priority_fn_v0 (worse), priority_fn_v1 (better). 
LLM is asked to write the next version (priority_fn_v2) but does not see scores 
(but performance ordering of priority_fn_v0 (worse) and priority_fn_v1 (better) could be a signal)

Note: 
- On the first call only Best Fit exists which is fed as priority_fn_v0 - LLM generates next version (priority_fn_v1)
- Any valid candidate so far can be a parent (including Best Fit)

Method A (independent sampling) uses the same function with only the starting program.
"""

import re

from src.prompts.method_a import SPECIFICATION
#same starting prompt as Method A

def generate_prompt(parents: list[str]) -> str:
    """parents: priority_fn sources, sorted from worse to better."""
    k = len(parents)
    new_name = f"priority_fn_v{k}"

    versions = [
        re.sub(r"\bpriority_fn\b", f"priority_fn_v{i}", code.strip())
        for i, code in enumerate(parents)
    ]
    header = (
        f"def {new_name}(item_size, open_bin_remaining_capacities, capacity):\n"
        f'    """Improved version of `priority_fn_v{k - 1}`."""'
    )
    if k == 1: #starting: BestFit is the only parent - evolve from bestfit
        intro = (
            "Below is a previously discovered priority function for online\n"
            "one-dimensional bin packing. Study it and develop a new heuristic\n"
            "that may improve on it."
        )
    else: 
        intro = (
            "Below are previously discovered priority functions for online\n"
            "one-dimensional bin packing, ordered from lower to higher\n"
            "performance. Study them and develop a new heuristic that may\n"
            "improve on them."
        )
    intro += (
        " You are not given any numerical evaluation results.\n"
        "Prioritise robust behaviour across differing distributions, sequence\n"
        "lengths, and item orderings, as required by the specification."
    )
    closing = (
        f"Complete {new_name}, using the same three arguments and returning\n"
        "the same score format. You may combine useful patterns from the\n"
        "versions above or introduce a different deterministic scoring rule.\n"
        f"Do not call earlier versions; {new_name} must stand alone.\n\n"
        f"Return only the complete Python definition of {new_name}.\n"
        "No Markdown fences, explanations, or additional functions."
    )
    return "\n\n".join([SPECIFICATION, intro, *versions, header, closing])
