"""
nsga2_feature_search.py

Defensive, offline NSGA-II optimizer for abstract feature-space robustness
experiments. This module does NOT read, write, patch, or mutate executable
files. It optimizes bounded numeric feature vectors supplied by the caller.

Objectives (all minimized):
  1. LightGBM malware confidence (or another caller-defined risk score)
  2. Siamese-SVM malware-vs-benign margin/risk score
  3. Normalized feature-space modification cost

The caller supplies `evaluate_fn(candidate) -> (lgb_risk, svm_risk)`.
Keep the evaluator deterministic and ensure its model/support data are drawn
from the permitted development partitions. Do not tune on the held-out test set.

Example:
    import numpy as np
    from nsga2_feature_search import NSGA2Config, run_nsga2

    lower = np.zeros(12)
    upper = np.ones(12)

    def evaluator(x):
        # Replace with your offline model-interface calls.
        # Both values must be risks where lower means less malware evidence.
        return float(...), float(...)

    result = run_nsga2(
        baseline=np.full(12, 0.5),
        lower_bounds=lower,
        upper_bounds=upper,
        evaluate_fn=evaluator,
        config=NSGA2Config(population_size=20, generations=5, seed=42),
    )

    for item in result["pareto_front"]:
        print(item["objectives"], item["candidate"])

This is a generic optimization component, not a claim that arbitrary
feature-space points correspond to realizable binaries or assembly programs.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Callable, Dict, List, Sequence, Tuple, Any
import json
import math
import numpy as np


ObjectiveFn = Callable[[np.ndarray], Tuple[float, float]]


@dataclass(frozen=True)
class NSGA2Config:
    population_size: int = 20
    generations: int = 5
    crossover_probability: float = 0.90
    mutation_probability: float = 0.15  # per dimension
    mutation_sigma: float = 0.08        # fraction of each dimension's range
    eta_c: float = 15.0                 # SBX distribution index
    eta_m: float = 20.0                 # polynomial mutation index
    seed: int = 42


def _validate_inputs(
    baseline: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    cfg: NSGA2Config,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(baseline, dtype=float).reshape(-1)
    lo = np.asarray(lower, dtype=float).reshape(-1)
    hi = np.asarray(upper, dtype=float).reshape(-1)

    if x.size == 0 or lo.shape != x.shape or hi.shape != x.shape:
        raise ValueError("baseline/lower_bounds/upper_bounds must be non-empty vectors of equal length")
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))):
        raise ValueError("baseline and bounds must contain only finite values")
    if np.any(lo >= hi):
        raise ValueError("Every lower bound must be strictly less than its upper bound")
    if np.any(x < lo) or np.any(x > hi):
        raise ValueError("baseline must lie within the supplied bounds")
    if cfg.population_size < 4 or cfg.population_size % 2:
        raise ValueError("population_size must be an even integer >= 4")
    if cfg.generations < 1:
        raise ValueError("generations must be >= 1")
    for name, value in (
        ("crossover_probability", cfg.crossover_probability),
        ("mutation_probability", cfg.mutation_probability),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")
    if cfg.mutation_sigma < 0:
        raise ValueError("mutation_sigma must be non-negative")
    return x, lo, hi


def _evaluate(
    candidate: np.ndarray,
    baseline: np.ndarray,
    span: np.ndarray,
    evaluate_fn: ObjectiveFn,
) -> Tuple[float, float, float]:
    a, b = evaluate_fn(candidate.copy())
    a, b = float(a), float(b)
    if not (math.isfinite(a) and math.isfinite(b)):
        raise ValueError("evaluate_fn returned a non-finite objective")
    # Mean normalized L1 distance. This is a cost objective, not a claim of
    # semantic validity or realizability.
    cost = float(np.mean(np.abs(candidate - baseline) / span))
    return a, b, cost


def _dominates(a: Sequence[float], b: Sequence[float]) -> bool:
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


def _fast_non_dominated_sort(pop: List[Dict[str, Any]]) -> List[List[int]]:
    dominates_set = [[] for _ in pop]
    dominated_count = [0 for _ in pop]
    fronts: List[List[int]] = [[]]

    for p in range(len(pop)):
        for q in range(len(pop)):
            if p == q:
                continue
            if _dominates(pop[p]["objectives"], pop[q]["objectives"]):
                dominates_set[p].append(q)
            elif _dominates(pop[q]["objectives"], pop[p]["objectives"]):
                dominated_count[p] += 1
        if dominated_count[p] == 0:
            pop[p]["rank"] = 0
            fronts[0].append(p)

    i = 0
    while i < len(fronts) and fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in dominates_set[p]:
                dominated_count[q] -= 1
                if dominated_count[q] == 0:
                    pop[q]["rank"] = i + 1
                    next_front.append(q)
        if next_front:
            fronts.append(next_front)
        i += 1
    return fronts


def _crowding_distance(pop: List[Dict[str, Any]], front: List[int]) -> None:
    for idx in front:
        pop[idx]["crowding"] = 0.0
    if len(front) <= 2:
        for idx in front:
            pop[idx]["crowding"] = float("inf")
        return

    n_obj = len(pop[front[0]]["objectives"])
    for m in range(n_obj):
        ordered = sorted(front, key=lambda idx: pop[idx]["objectives"][m])
        pop[ordered[0]]["crowding"] = float("inf")
        pop[ordered[-1]]["crowding"] = float("inf")
        lo = pop[ordered[0]]["objectives"][m]
        hi = pop[ordered[-1]]["objectives"][m]
        if hi == lo:
            continue
        for j in range(1, len(ordered) - 1):
            idx = ordered[j]
            if math.isinf(pop[idx]["crowding"]):
                continue
            prev_v = pop[ordered[j - 1]]["objectives"][m]
            next_v = pop[ordered[j + 1]]["objectives"][m]
            pop[idx]["crowding"] += (next_v - prev_v) / (hi - lo)


def _environmental_selection(
    combined: List[Dict[str, Any]], target_size: int
) -> List[Dict[str, Any]]:
    fronts = _fast_non_dominated_sort(combined)
    selected: List[Dict[str, Any]] = []
    for front in fronts:
        _crowding_distance(combined, front)
        if len(selected) + len(front) <= target_size:
            selected.extend(combined[i] for i in front)
        else:
            ordered = sorted(
                front,
                key=lambda i: combined[i]["crowding"],
                reverse=True,
            )
            selected.extend(combined[i] for i in ordered[: target_size - len(selected)])
            break
    return selected


def _tournament(pop: List[Dict[str, Any]], rng: np.random.Generator) -> Dict[str, Any]:
    i, j = rng.integers(0, len(pop), size=2)
    a, b = pop[int(i)], pop[int(j)]
    if a["rank"] < b["rank"]:
        return a
    if b["rank"] < a["rank"]:
        return b
    return a if a["crowding"] >= b["crowding"] else b


def _sbx(
    p1: np.ndarray,
    p2: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    eta: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    c1, c2 = p1.copy(), p2.copy()
    for i in range(p1.size):
        if rng.random() > 0.5 or abs(p1[i] - p2[i]) < 1e-14:
            continue
        x1, x2 = sorted((p1[i], p2[i]))
        rand = rng.random()

        beta = 1.0 + (2.0 * (x1 - lo[i]) / (x2 - x1))
        alpha = 2.0 - beta ** (-(eta + 1.0))
        if rand <= 1.0 / alpha:
            betaq = (rand * alpha) ** (1.0 / (eta + 1.0))
        else:
            betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1.0))
        child1 = 0.5 * ((x1 + x2) - betaq * (x2 - x1))

        beta = 1.0 + (2.0 * (hi[i] - x2) / (x2 - x1))
        alpha = 2.0 - beta ** (-(eta + 1.0))
        if rand <= 1.0 / alpha:
            betaq = (rand * alpha) ** (1.0 / (eta + 1.0))
        else:
            betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1.0))
        child2 = 0.5 * ((x1 + x2) + betaq * (x2 - x1))

        if rng.random() <= 0.5:
            c1[i], c2[i] = child2, child1
        else:
            c1[i], c2[i] = child1, child2

    return np.clip(c1, lo, hi), np.clip(c2, lo, hi)


def _polynomial_mutation(
    x: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    probability: float,
    eta: float,
    sigma_fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    y = x.copy()
    for i in range(y.size):
        if rng.random() >= probability:
            continue
        span = hi[i] - lo[i]
        delta1 = (y[i] - lo[i]) / span
        delta2 = (hi[i] - y[i]) / span
        rand = rng.random()
        mut_pow = 1.0 / (eta + 1.0)
        if rand < 0.5:
            xy = 1.0 - delta1
            val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (eta + 1.0))
            deltaq = (val ** mut_pow) - 1.0
        else:
            xy = 1.0 - delta2
            val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (eta + 1.0))
            deltaq = 1.0 - (val ** mut_pow)
        # sigma_fraction controls the maximum practical step scale.
        y[i] += deltaq * span * min(1.0, sigma_fraction * 4.0)
    return np.clip(y, lo, hi)


def run_nsga2(
    baseline: Sequence[float],
    lower_bounds: Sequence[float],
    upper_bounds: Sequence[float],
    evaluate_fn: ObjectiveFn,
    config: NSGA2Config = NSGA2Config(),
) -> Dict[str, Any]:
    """
    Run NSGA-II around one baseline feature vector.

    `evaluate_fn` must return two scalar risks, both minimized. The third
    objective (normalized L1 modification cost) is calculated internally.
    The baseline is retained in the initial population.

    Returns a dict with `pareto_front`, `history`, `config`, and `baseline`.
    """
    base, lo, hi = _validate_inputs(
        np.asarray(baseline), np.asarray(lower_bounds), np.asarray(upper_bounds), config
    )
    rng = np.random.default_rng(config.seed)
    span = hi - lo
    n = config.population_size

    def individual(x: np.ndarray) -> Dict[str, Any]:
        x = np.clip(np.asarray(x, dtype=float), lo, hi)
        return {
            "candidate": x,
            "objectives": _evaluate(x, base, span, evaluate_fn),
            "rank": None,
            "crowding": 0.0,
        }

    population = [individual(base)]
    for _ in range(n - 1):
        population.append(individual(rng.uniform(lo, hi)))

    history = []
    for generation in range(config.generations + 1):
        fronts = _fast_non_dominated_sort(population)
        for front in fronts:
            _crowding_distance(population, front)
        first = [population[i] for i in fronts[0]]
        objectives = np.asarray([p["objectives"] for p in first])
        history.append({
            "generation": generation,
            "population_size": len(population),
            "pareto_size": len(first),
            "best_lgb_risk": float(objectives[:, 0].min()),
            "best_svm_risk": float(objectives[:, 1].min()),
            "best_cost": float(objectives[:, 2].min()),
        })
        if generation == config.generations:
            break

        offspring = []
        while len(offspring) < n:
            p1 = _tournament(population, rng)["candidate"]
            p2 = _tournament(population, rng)["candidate"]
            if rng.random() < config.crossover_probability:
                c1, c2 = _sbx(p1, p2, lo, hi, config.eta_c, rng)
            else:
                c1, c2 = p1.copy(), p2.copy()
            c1 = _polynomial_mutation(
                c1, lo, hi, config.mutation_probability, config.eta_m,
                config.mutation_sigma, rng
            )
            c2 = _polynomial_mutation(
                c2, lo, hi, config.mutation_probability, config.eta_m,
                config.mutation_sigma, rng
            )
            offspring.append(individual(c1))
            if len(offspring) < n:
                offspring.append(individual(c2))

        # Correct NSGA-II environmental selection: parents + offspring (2N).
        population = _environmental_selection(population + offspring, n)

    final_fronts = _fast_non_dominated_sort(population)
    _crowding_distance(population, final_fronts[0])
    pareto = []
    for idx in final_fronts[0]:
        p = population[idx]
        pareto.append({
            "candidate": p["candidate"].tolist(),
            "objectives": list(p["objectives"]),
            "rank": int(p["rank"]),
            "crowding": p["crowding"] if math.isfinite(p["crowding"]) else "inf",
        })

    return {
        "baseline": base.tolist(),
        "bounds": {"lower": lo.tolist(), "upper": hi.tolist()},
        "config": asdict(config),
        "history": history,
        "pareto_front": pareto,
        "notes": [
            "All objectives are minimized.",
            "Feature-space candidates are abstract and are not guaranteed to be realizable.",
            "No executable files are read, written, or modified.",
            "Use validation for development and reserve test data for final evaluation.",
        ],
    }


def save_result(result: Dict[str, Any], output_path: str) -> None:
    """Save JSON-compatible optimization results."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, allow_nan=False)
