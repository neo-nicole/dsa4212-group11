
from src.prompts.method_a import INITIAL_PROMPT


EVOLUTION_PROMPT = """
Below are two previously discovered priority functions for online
one-dimensional bin packing, ordered from lower to higher performance.
Study both implementations and develop a new heuristic that may improve
on them. You are not given their numerical evaluation results.
Prioritise robust behaviour across differing distributions, sequence
lengths, and item orderings, as required by the shared specification.

{candidate_v0}

{candidate_v1}

Write the next version, priority_v2, using the same three arguments and
returning the same score format. You may combine useful patterns from
both examples or introduce a different deterministic scoring rule.
Do not call priority_v0 or priority_v1; your function must stand alone.

Return only the complete Python definition of priority_v2.
No Markdown fences, explanations, or additional functions.
""".strip()

def build_evolution_prompt(candidate_v0: str, candidate_v1: str) -> str:
    """Construct best-shot prompt from two already-versioned definitions.

    candidate_v0 must define priority_v0 (lower-ranked), and
    candidate_v1 must define priority_v1 (higher-ranked).
    Ranking is determined by the search controller, not the LLM.
    """
    return EVOLUTION_PROMPT.format(
        candidate_v0=candidate_v0.strip(),
        candidate_v1=candidate_v1.strip(),
    )
