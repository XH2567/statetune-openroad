# src/ranktuner_utils.py
# MEMORY SAFETY: Cap the number of pairwise comparisons to avoid O(n^2) growth.
import numpy as np
import torch
from typing import Dict

def is_pareto_dominated(y1, y2):
    """
    Return True if y1 is dominated by y2 (that is, y2 is better).

    Assumed metric order: [Area(min), WNS(max), Power(min)].

    We normalize the objectives into a minimization form:
    y_norm = [Area, -WNS, Power]
    """
    # Convert WNS from maximize to minimize by negating it.
    # Area and Power are already "smaller is better".
    obj1 = np.array([y1['area'], -y1['wns'], y1['power']])
    obj2 = np.array([y2['area'], -y2['wns'], y2['power']])

    # y2 dominates y1 iff:
    # - y2 is no worse than y1 on every objective, and
    # - y2 is strictly better on at least one objective.
    dominates = np.all(obj2 <= obj1) and np.any(obj2 < obj1)
    return dominates

def is_pareto_dominated(a: Dict[str, float], b: Dict[str, float]) -> bool:
    # Minimize area/power, maximize WNS.
    return (b["area"] <= a["area"] and b["power"] <= a["power"] and b["wns"] >= a["wns"]) and (
        b["area"] < a["area"] or b["power"] < a["power"] or b["wns"] > a["wns"]
    )

def is_valid(y):
    """Return True if the result is valid (that is, not a failed run)."""
    return y['area'] != float('inf')

def generate_comparisons(history, max_comps: int = 5000):
    """
    Generate pairwise comparison indices.

    MEMORY SAFETY:
    When the history becomes large, only the most recent window of records is
    used so the number of comparisons does not grow as O(n^2).

    Example comparison counts:
      n=100 -> 4,950 pairs
      n=200 -> 19,900 pairs
      n=500 -> 124,750 pairs

    The effective window size is capped by ``max_comps``.
    """
    n = len(history)
    if n < 2:
        return torch.tensor([], dtype=torch.long)

    # If history is too long, keep only the most recent window.
    if n * (n - 1) // 2 > max_comps:
        import math
        window = int((1 + math.sqrt(1 + 8 * max_comps)) / 2)
        history = history[-window:]
        n = len(history)

    comps = []
    for i in range(n):
        for j in range(i + 1, n):
            y1 = history[i]['metrics']
            y2 = history[j]['metrics']

            valid_1 = is_valid(y1)
            valid_2 = is_valid(y2)

            # Case 1: both runs succeeded -> compare by Pareto dominance.
            if valid_1 and valid_2:
                if is_pareto_dominated(y1, y2):  # y2 is better
                    comps.append([j, i])
                elif is_pareto_dominated(y2, y1):  # y1 is better
                    comps.append([i, j])

            # Case 2: one succeeded and one failed -> the successful run wins.
            elif valid_1 and not valid_2:
                comps.append([i, j])  # i (success) beats j (failure)
            elif not valid_1 and valid_2:
                comps.append([j, i])  # j (success) beats i (failure)

            # Case 3: both failed -> ignore, no comparison is generated.

    if not comps:
        return torch.tensor([], dtype=torch.long)

    return torch.tensor(comps, dtype=torch.long)
