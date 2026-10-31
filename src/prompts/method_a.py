
INITIAL_PROMPT = """
You are an expert Python programmer and optimisation researcher
specialising in online bin packing heuristics. Your task is to design
a deterministic bin-selection heuristic for the one-dimensional
online bin packing problem.

PROBLEM DEFINITION

Items arrive sequentially and future item sizes are unknown.
- Bin capacity: 100
- Item sizes: integers from 1 to 100
- Every item must be packed.
- No bin may exceed its capacity.
- The objective is to minimise the total number of bins used.

REQUIRED FUNCTION

Implement exactly this Python function:

def priority_fn(item_size, open_bin_remaining_capacities, capacity):

INPUTS
- item_size: Size of the current item.
- open_bin_remaining_capacities: A NumPy array or list containing
  the remaining capacities of currently open bins that can fit
  the current item. Bins that cannot fit the item have already
  been removed. The array may be empty.
- capacity: Maximum bin capacity.

OUTPUTS
- Return a list of numerical priority scores.
- The list must have length
  len(open_bin_remaining_capacities) + 1.
- Score at index i corresponds to placing the item into existing
  bin i.
- The final score corresponds to opening a new bin.
- Higher scores indicate more preferred actions.
- The simulator selects the feasible action with the highest score.

CONSTRAINTS
1. Implement only the priority_fn function.
2. Use only the current item size, existing bin capacities,
   and total bin capacity.
3. Do not use future items or information about the full input sequence.
4. Do not use external libraries, file access, randomness,
   or global variables.
5. The function must be deterministic and computationally efficient.
6. Return only finite numerical scores (no NaN or infinity).
7. Always return the correct number of scores.
   If the array is empty, return exactly one score for opening a new bin.
8. The new-bin action must always remain feasible.

OUTPUT FORMAT

Return only valid Python code containing the complete priority_fn
function. Do not include Markdown code fences, explanations,
examples, or additional functions.
""".strip()
