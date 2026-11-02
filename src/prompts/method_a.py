
INITIAL_PROMPT = """
You are an expert Python programmer and optimisation researcher
specialising in online one-dimensional bin packing heuristics.

PROBLEM DEFINITION
Items arrive sequentially; future item sizes are unknown.
- Bin capacity: 100.
- Item sizes: integers from 1 to 100.
- Pack every item without exceeding bin capacity.
- Minimise the number of bins used.

FUNCTION INTERFACE
Implement a deterministic Python function:

def priority_fn(item_size, open_bin_remaining_capacities, capacity):

- item_size: current item size.
- open_bin_remaining_capacities: a list or NumPy array of remaining
  capacities for ONLY existing bins that can fit the current item.
  It may be empty.
- capacity: maximum bin capacity.
- Return len(open_bin_remaining_capacities) + 1 finite numerical scores.
- Scores for existing bins come first, in their given order.
- The final score represents opening a new bin.
- The simulator selects the action with the HIGHEST score.

GENERALISATION OBJECTIVE
Design a general-purpose heuristic that remains effective across item-size
 distributions, sequence lengths, and arrival orderings different from
 those encountered during development. Avoid relying on a particular
 distribution, sequence length, or ordering. Decisions must use only
 the current item and feasible existing-bin capacities; never assume
 knowledge of future items or the item-generation process.

CONSTRAINTS
- Use only the three function arguments; never inspect future items.
- Do not import any libraries. NumPy is already available as np.
- You may use NumPy operations supported by the sandbox.
- Use only operations permitted by the execution environment.
- Do not use file access, randomness, or mutable global state.
- Do not define helper functions.
- Return exactly one score if there are no feasible existing bins.
- Keep the implementation deterministic and computationally efficient.
- Do not modify the simulator or evaluator.

OUTPUT
Return only valid Python code for one complete function.
No Markdown fences, explanations, or additional functions.
""".strip()
