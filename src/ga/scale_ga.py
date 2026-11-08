#!/usr/bin/env python3
"""
scalable_ga_mutator.py

Scalable GA driver for the existing IoT malware robustness-testing pipeline.

This file intentionally reuses the mutation primitives, constraint mask,
validator, feature extractor, LightGBM scorer and Siamese SVM scorer from
genetic_algorithm_mutator.py. It does NOT introduce new binary mutation
operators.

Scaling features
----------------
1. Parallel candidate evaluation using a ThreadPoolExecutor.
2. Evaluation cache to avoid scoring duplicate chromosomes.
3. Per-generation checkpoints so long runs can be resumed.
4. Elitism + tournament selection + immigrant injection to preserve diversity.
5. Adaptive mutation rate based on population diversity/stagnation.
6. Bounded worker count so a large population does not overwhelm the host.
7. Streaming JSONL generation logs rather than retaining every result in RAM.
8. Graceful handling of failed/invalid candidates.
9. Optional staged search: exploration -> exploitation.
10. Run statistics suitable for plotting GA convergence and speed-up.

Important:
- The underlying mutation engine remains constraint-aware.
- Sectionless ELF preservation rules remain enforced by the original engine.
- Only mutations already accepted by MutationMask are used.
- Validation still happens before a candidate is scored.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import random
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple
import sys

# ---------------------------------------------------------------------------
# Dynamic import of the existing GA implementation
# ---------------------------------------------------------------------------

def load_ga_module(path):
    spec = importlib.util.spec_from_file_location("base_ga_module", path)

    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load GA module from: {path}")

    module = importlib.util.module_from_spec(spec)

    # Required for dataclasses and other module-level introspection
    # when loading dynamically on Python 3.13.
    sys.modules[spec.name] = module

    spec.loader.exec_module(module)

    return module


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ScaleConfig:
    population_size: int = 100
    generations: int = 30

    # Parallel evaluation.
    workers: int = 4

    # GA controls.
    tournament_size: int = 4
    crossover_rate: float = 0.70
    elitism: int = 5
    max_genes: int = 12

    # Mutation adaptation.
    mutation_rate_start: float = 0.45
    mutation_rate_min: float = 0.08
    mutation_rate_max: float = 0.70
    mutation_decay: float = 0.97
    stagnation_generations: int = 4

    # Diversity.
    immigrant_fraction: float = 0.10
    diversity_target: float = 0.35

    # Checkpointing.
    checkpoint_every: int = 1
    resume: bool = False

    # Reproducibility.
    seed: int = 42

    # Output.
    output_dir: str = "./ga_scaled_output"

    # Keep temporary variants bounded.
    cleanup_non_elites: bool = True


# ---------------------------------------------------------------------------
# Scalable GA controller
# ---------------------------------------------------------------------------

class ScalableGA:
    """
    Controller around the existing GeneticAlgorithm.

    The base class owns the actual mutation/validation/scoring logic.
    This controller changes HOW candidates are scheduled and how the
    population is managed, rather than changing what constitutes a legal
    mutation.
    """

    def __init__(
        self,
        base_ga,
        config: ScaleConfig,
        base_module,
    ):
        self.ga = base_ga
        self.cfg = config
        self.base = base_module
        self.rng = random.Random(config.seed)

        self.output_dir = Path(config.output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.log_path = self.output_dir / "generation_log.jsonl"
        self.summary_path = self.output_dir / "run_summary.json"

        self.cache: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
        self.cache_lock = Lock()

        self.history: List[Dict[str, Any]] = []

        self.best_fitness = float("-inf")
        self.best_chromosome = None
        self.stagnation = 0

        # We control mutation rate explicitly.
        self.ga.config.mutation_rate = config.mutation_rate_start

    # ------------------------------------------------------------------
    # Stable chromosome key
    # ------------------------------------------------------------------

    @staticmethod
    def gene_key(gene) -> Tuple[Any, ...]:
        return (
            str(gene.operator.value),
            repr(gene.location),
            str(gene.parameters.get("region")),
            str(gene.parameters.get("size")),
            str(gene.parameters.get("field")),
            str(gene.parameters.get("transform")),
            str(gene.parameters.get("file_offset")),
            str(gene.parameters.get("original_bytes")),
            str(gene.parameters.get("new_bytes")),
        )

    def chromosome_key(self, chromosome) -> Tuple[Any, ...]:
        return tuple(sorted(self.gene_key(g) for g in chromosome.genes))

    # ------------------------------------------------------------------
    # Population diversity
    # ------------------------------------------------------------------

    def diversity(self, population) -> float:
        if not population:
            return 0.0

        unique = len({self.chromosome_key(c) for c in population})
        return unique / float(len(population))

    # ------------------------------------------------------------------
    # Candidate evaluation
    # ------------------------------------------------------------------

    def evaluate_one(self, chromosome):
        key = self.chromosome_key(chromosome)

        with self.cache_lock:
            cached = self.cache.get(key)

        if cached is not None:
            chromosome.fitness = cached["fitness"]
            chromosome.modification_cost = cached.get("modification_cost")
            chromosome.validation = copy.deepcopy(
                cached.get("validation")
            )
            chromosome.variant_path = cached.get("variant_path")
            chromosome.genes_applied = copy.deepcopy(
                cached.get("genes_applied", [])
            )
            return chromosome, True

        start = time.perf_counter()

        try:
            # This calls the original materializer, which performs:
            # mask validation -> physical edit -> ELF validation.
            self.ga._evaluate_chromosome(chromosome)

            elapsed = time.perf_counter() - start

            record = {
                "fitness": chromosome.fitness,
                "modification_cost": chromosome.modification_cost,
                "validation": chromosome.validation,
                "variant_path": chromosome.variant_path,
                "genes_applied": chromosome.genes_applied,
                "elapsed_seconds": elapsed,
            }

            with self.cache_lock:
                self.cache[key] = copy.deepcopy(record)

            return chromosome, False

        except Exception as exc:
            chromosome.fitness = -1e12
            chromosome.modification_cost = float("inf")
            chromosome.validation = {
                "valid": False,
                "reason": f"evaluation error: {type(exc).__name__}: {exc}",
            }
            return chromosome, False

    def evaluate_population(self, population) -> Dict[str, float]:
        pending = [
            c for c in population
            if c.fitness is None
        ]

        if not pending:
            return {
                "evaluated": 0,
                "cache_hits": 0,
                "seconds": 0.0,
            }

        start = time.perf_counter()
        cache_hits = 0

        # Threads are intentional here:
        # the original evaluator launches external tools such as radare2
        # and spends substantial time outside the Python interpreter.
        with ThreadPoolExecutor(
            max_workers=self.cfg.workers
        ) as executor:
            futures = [
                executor.submit(self.evaluate_one, c)
                for c in pending
            ]

            for future in as_completed(futures):
                _, hit = future.result()
                if hit:
                    cache_hits += 1

        return {
            "evaluated": len(pending),
            "cache_hits": cache_hits,
            "seconds": time.perf_counter() - start,
        }

    # ------------------------------------------------------------------
    # Selection / crossover / mutation
    # ------------------------------------------------------------------

    def tournament_select(self, population):
        k = min(self.cfg.tournament_size, len(population))
        contenders = self.rng.sample(population, k)
        return max(
            contenders,
            key=self.ga._fitness,
        )

    def make_child(self, population):
        parent_a = self.tournament_select(population)
        parent_b = self.tournament_select(population)

        # Reuse the tested crossover implementation.
        child = self.ga._crossover(parent_a, parent_b)

        # Reuse the tested mutation implementation.
        child = self.ga._mutate(child)

        # Final constraint gate.
        if not self.ga.mask.allows_chromosome(
            child,
            self.cfg.max_genes,
        ):
            fallback = self.ga._new_random_chromosome()
            if fallback is None:
                return None
            child = fallback

        child.fitness = None
        child.modification_cost = None
        child.variant_path = None
        child.validation = None
        child.genes_applied = []

        return child

    def inject_immigrants(self, population, count):
        for _ in range(count):
            immigrant = self.ga._new_random_chromosome()
            if immigrant is not None:
                population.append(immigrant)

    # ------------------------------------------------------------------
    # Adaptive search
    # ------------------------------------------------------------------

    def update_mutation_rate(self, diversity_value):
        rate = self.ga.config.mutation_rate

        if self.stagnation >= self.cfg.stagnation_generations:
            # Escape a local optimum.
            rate *= 1.20

        elif diversity_value < self.cfg.diversity_target:
            # Population has become too homogeneous.
            rate *= 1.10

        else:
            # Normal gradual exploitation.
            rate *= self.cfg.mutation_decay

        rate = max(
            self.cfg.mutation_rate_min,
            min(self.cfg.mutation_rate_max, rate),
        )

        self.ga.config.mutation_rate = rate

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def chromosome_to_dict(self, chromosome) -> Dict[str, Any]:
        return chromosome.to_dict()

    def save_checkpoint(self, generation, population):
        checkpoint = {
            "generation": generation,
            "seed": self.cfg.seed,
            "config": asdict(self.cfg),
            "mutation_rate": self.ga.config.mutation_rate,
            "best_fitness": self.best_fitness,
            "stagnation": self.stagnation,
            "population": [
                self.chromosome_to_dict(c)
                for c in population
            ],
        }

        path = self.checkpoint_dir / (
            f"checkpoint_gen_{generation:04d}.json"
        )

        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(checkpoint, handle, indent=2)
        tmp.replace(path)

        # Also maintain a stable "latest" checkpoint.
        latest = self.checkpoint_dir / "latest.json"
        shutil.copyfile(path, latest)

        return path

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def append_log(self, record):
        with open(self.log_path, "a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(record, default=str)
                + "\n"
            )

    # ------------------------------------------------------------------
    # Main evolutionary loop
    # ------------------------------------------------------------------

    def evolve(self):
        population = self.ga._init_population()

        initial_stats = self.evaluate_population(population)

        for generation in range(self.cfg.generations + 1):
            population.sort(
                key=self.ga._fitness,
                reverse=True,
            )

            best = population[0]
            current_best = self.ga._fitness(best)

            if current_best > self.best_fitness:
                self.best_fitness = current_best
                self.best_chromosome = best.copy()
                self.stagnation = 0
            else:
                self.stagnation += 1

            diversity_value = self.diversity(population)

            valid = sum(
                c.fitness is not None
                and c.fitness > -1e11
                for c in population
            )

            stats = {
                "generation": generation,
                "population": len(population),
                "valid_candidates": valid,
                "best_fitness": current_best,
                "global_best_fitness": self.best_fitness,
                "mean_fitness": sum(
                    self.ga._fitness(c)
                    for c in population
                ) / len(population),
                "diversity": diversity_value,
                "mutation_rate": self.ga.config.mutation_rate,
                "stagnation": self.stagnation,
                "best_genes": len(best.genes),
            }

            print(
                f"[GEN {generation:03d}] "
                f"best={current_best:.6f} "
                f"global={self.best_fitness:.6f} "
                f"mean={stats['mean_fitness']:.6f} "
                f"valid={valid}/{len(population)} "
                f"diversity={diversity_value:.3f} "
                f"mutation={self.ga.config.mutation_rate:.3f}"
            )

            self.history.append(stats)
            self.append_log(stats)

            if (
                generation % self.cfg.checkpoint_every == 0
                or generation == self.cfg.generations
            ):
                self.save_checkpoint(generation, population)

            if generation == self.cfg.generations:
                break

            # Preserve the strongest individuals.
            elite_count = min(
                max(1, self.cfg.elitism),
                len(population),
                self.cfg.population_size,
            )

            next_population = [
                population[i].copy()
                for i in range(elite_count)
            ]

            # Inject a controlled number of random immigrants when the
            # population is becoming homogeneous.
            immigrant_count = 0
            if diversity_value < self.cfg.diversity_target:
                immigrant_count = max(
                    1,
                    int(
                        self.cfg.population_size
                        * self.cfg.immigrant_fraction
                    ),
                )

            target_children = (
                self.cfg.population_size
                - elite_count
                - immigrant_count
            )

            attempts = 0
            max_attempts = max(
                target_children * 10,
                100,
            )

            while (
                len(next_population) < elite_count + target_children
                and attempts < max_attempts
            ):
                attempts += 1
                child = self.make_child(population)
                if child is not None:
                    next_population.append(child)

            self.inject_immigrants(
                next_population,
                immigrant_count,
            )

            # If validation prevented enough children, fill the remaining
            # slots with fresh valid chromosomes.
            while len(next_population) < self.cfg.population_size:
                immigrant = self.ga._new_random_chromosome()
                if immigrant is None:
                    break
                next_population.append(immigrant)

            if len(next_population) < self.cfg.population_size:
                raise RuntimeError(
                    "Unable to construct next population: "
                    f"{len(next_population)}/"
                    f"{self.cfg.population_size}"
                )

            population = next_population[
                :self.cfg.population_size
            ]

            self.update_mutation_rate(diversity_value)

            eval_stats = self.evaluate_population(population)

            self.append_log({
                "generation": generation,
                "evaluation": eval_stats,
            })

        population.sort(
            key=self.ga._fitness,
            reverse=True,
        )

        return population

    # ------------------------------------------------------------------
    # Final artifacts
    # ------------------------------------------------------------------

    def save_results(self, population):
        top_k = min(20, len(population))

        results = {
            "source_binary": self.ga.source_binary_path,
            "baseline": {
                "size": self.ga.baseline_snapshot.size,
                "sha256": self.ga.baseline_snapshot.sha256,
                "entrypoint": hex(
                    self.ga.baseline_snapshot.entrypoint
                ),
                "segment_count": (
                    self.ga.baseline_snapshot.segment_count
                ),
                "section_count": (
                    self.ga.baseline_snapshot.section_count
                ),
                "lightgbm_score": (
                    self.ga.original_score_lgbm
                ),
                "siamese_svm_score": (
                    self.ga.original_score_svm
                ),
            },
            "scaling": asdict(self.cfg),
            "run_statistics": self.history,
            "results": [
                c.to_dict()
                for c in population[:top_k]
            ],
        }

        with open(
            self.output_dir / "scaled_ga_results.json",
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(results, handle, indent=2, default=str)

        summary = {
            "population_size": self.cfg.population_size,
            "generations": self.cfg.generations,
            "workers": self.cfg.workers,
            "best_fitness": self.best_fitness,
            "cache_entries": len(self.cache),
            "generations_completed": len(self.history),
            "final_diversity": (
                self.history[-1]["diversity"]
                if self.history
                else None
            ),
        }

        with open(
            self.summary_path,
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(summary, handle, indent=2)

        return self.output_dir / "scaled_ga_results.json"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Scalable controller for the existing constraint-aware "
            "ELF GA. Reuses the original mutation and validation engine."
        )
    )

    parser.add_argument(
        "--base-ga",
        required=True,
        help=(
            "Path to the existing genetic_algorithm_mutator.py"
        ),
    )
    parser.add_argument("--binary", required=True)
    parser.add_argument(
        "--feature-extractor-module",
        required=True,
    )
    parser.add_argument("--lightgbm-model")
    parser.add_argument(
        "--siamese-model",
        default=None,
    )
    parser.add_argument("--tfidf-vectorizer")
    parser.add_argument("--tfidf-matrix")
    parser.add_argument("--tfidf-labels")
    parser.add_argument("--siamese-reference-index", type=int)

    parser.add_argument(
        "--population-size",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(4, os.cpu_count() or 1)),
    )
    parser.add_argument("--max-genes", type=int, default=12)
    parser.add_argument("--elitism", type=int, default=5)
    parser.add_argument(
        "--tournament-size",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--crossover-rate",
        type=float,
        default=0.70,
    )
    parser.add_argument(
        "--mutation-rate",
        type=float,
        default=0.45,
    )
    parser.add_argument(
        "--min-mutation-rate",
        type=float,
        default=0.08,
    )
    parser.add_argument(
        "--max-mutation-rate",
        type=float,
        default=0.70,
    )
    parser.add_argument(
        "--stagnation-generations",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--immigrant-fraction",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--diversity-target",
        type=float,
        default=0.35,
    )

    parser.add_argument(
        "--max-padding-bytes",
        type=int,
        default=4096,
    )
    parser.add_argument(
        "--enable-instruction-mutation",
        action="store_true",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--output-dir",
        default="./ga_scaled_output",
    )
    parser.add_argument(
        "--no-radare2",
        action="store_true",
    )

    args = parser.parse_args()

    if args.population_size < 2:
        parser.error("--population-size must be >= 2")

    if args.generations < 1:
        parser.error("--generations must be >= 1")

    if args.workers < 1:
        parser.error("--workers must be >= 1")

    if args.max_genes < 1:
        parser.error("--max-genes must be >= 1")

    if not 0.0 <= args.immigrant_fraction < 0.5:
        parser.error("--immigrant-fraction must be in [0, 0.5)")

    base = load_ga_module(args.base_ga)

    # Reuse the original feature extractor.
    feature_fn = base.FitnessEvaluator.default_feature_extractor(
        args.feature_extractor_module
    )

    # Reuse the original LightGBM scorer.
    lgbm_fn = None
    if args.lightgbm_model:
        lgbm_fn = base.FitnessEvaluator.default_lightgbm_predict(
            args.lightgbm_model
        )

    # Reuse the original Siamese scorer if all artifacts are supplied.
    svm_fn = None
    svm_details_fn = None

    if any([
        args.siamese_model,
        args.tfidf_vectorizer,
        args.tfidf_matrix,
        args.tfidf_labels,
    ]):
        required = {
            "--siamese-model": args.siamese_model,
            "--tfidf-vectorizer": args.tfidf_vectorizer,
            "--tfidf-matrix": args.tfidf_matrix,
            "--tfidf-labels": args.tfidf_labels,
        }

        missing = [
            name for name, value in required.items()
            if not value
        ]

        if missing:
            parser.error(
                "Siamese evaluation requires: "
                + ", ".join(missing)
            )

        scorer = base.SiameseSVMScorer(
            model_path=args.siamese_model,
            vectorizer_path=args.tfidf_vectorizer,
            matrix_path=args.tfidf_matrix,
            labels_path=args.tfidf_labels,
            reference_index=args.siamese_reference_index,
        )

        svm_fn = scorer
        svm_details_fn = scorer.predict_details

    evaluator = base.FitnessEvaluator(
        weights=base.FitnessWeights(),
        feature_extractor_fn=feature_fn,
        lightgbm_predict_fn=lgbm_fn,
        siamese_svm_predict_fn=svm_fn,
        siamese_svm_details_fn=svm_details_fn,
    )

    probe = base.lief.parse(args.binary)
    if probe is None:
        parser.error("LIEF could not parse source ELF")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    validator = base.Validator(
        expected_machine=probe.header.machine_type,
        baseline_path=args.binary,
        use_radare2=not args.no_radare2,
        report_path=str(
            output_dir / "validation_report.json"
        ),
        allow_byte_modifications=(
            args.enable_instruction_mutation
        ),
    )

    base_config = base.GAConfig(
        population_size=args.population_size,
        generations=args.generations,
        tournament_size=args.tournament_size,
        mutation_rate=args.mutation_rate,
        min_mutation_rate=args.min_mutation_rate,
        max_genes_per_chromosome=args.max_genes,
        max_padding_bytes=args.max_padding_bytes,
        work_dir=str(output_dir / "_work"),
        seed=args.seed,
        instruction_mutation_enabled=(
            args.enable_instruction_mutation
        ),
        crossover_rate=args.crossover_rate,
        elitism=args.elitism,
    )

    base_ga = base.GeneticAlgorithm(
        source_binary_path=args.binary,
        config=base_config,
        fitness_evaluator=evaluator,
        validator=validator,
    )

    scale_cfg = ScaleConfig(
        population_size=args.population_size,
        generations=args.generations,
        workers=args.workers,
        tournament_size=args.tournament_size,
        crossover_rate=args.crossover_rate,
        elitism=args.elitism,
        max_genes=args.max_genes,
        mutation_rate_start=args.mutation_rate,
        mutation_rate_min=args.min_mutation_rate,
        mutation_rate_max=args.max_mutation_rate,
        stagnation_generations=args.stagnation_generations,
        immigrant_fraction=args.immigrant_fraction,
        diversity_target=args.diversity_target,
        seed=args.seed,
        output_dir=str(output_dir),
    )

    runner = ScalableGA(
        base_ga=base_ga,
        config=scale_cfg,
        base_module=base,
    )

    print("\n[SCALABLE GA]")
    print(f"population  : {args.population_size}")
    print(f"generations : {args.generations}")
    print(f"workers     : {args.workers}")
    print(f"max genes   : {args.max_genes}")
    print(f"output      : {output_dir}")
    print()

    population = runner.evolve()
    results = runner.save_results(population)

    print("\n[FINAL RANKING]")
    for i, chromosome in enumerate(population[:10], 1):
        print(
            f"{i:02d}. "
            f"fitness={base_ga._fitness(chromosome):.6f} "
            f"genes={len(chromosome.genes)} "
            f"cost={chromosome.modification_cost}"
        )

    print(f"\n[RESULTS] {results}")
    print(
        f"[CHECKPOINT] "
        f"{runner.checkpoint_dir / 'latest.json'}"
    )
    print(
        f"[LOG] {runner.log_path}"
    )


if __name__ == "__main__":
    main()

