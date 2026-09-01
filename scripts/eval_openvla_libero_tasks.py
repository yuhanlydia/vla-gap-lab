#!/usr/bin/env python3
"""Run a small task subset through the official OpenVLA-OFT LIBERO evaluator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


class CounterfactualCleanSuite:
    """Use variant init states with its official unperturbed base BDDL."""

    def __init__(self, suite, task_id: int, bddl_file: str, language: str) -> None:
        self.suite = suite
        self.task_id = task_id
        self.bddl_file = bddl_file
        self.language = language

    def get_task(self, task_id: int):
        task = self.suite.get_task(task_id)
        if task_id != self.task_id:
            return task
        return task._replace(
            name=Path(self.bddl_file).stem,
            language=self.language,
            bddl_file=self.bddl_file,
        )

    def get_task_init_states(self, task_id: int):
        return self.suite.get_task_init_states(task_id)


class LanguageOverrideSuite:
    """Keep benchmark dynamics and visuals while replacing instruction text."""

    def __init__(self, suite, task_id: int, language: str) -> None:
        self.suite = suite
        self.task_id = task_id
        self.language = language

    def get_task(self, task_id: int):
        task = self.suite.get_task(task_id)
        if task_id != self.task_id:
            return task
        return task._replace(language=self.language)

    def get_task_init_states(self, task_id: int):
        return self.suite.get_task_init_states(task_id)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--task-ids", type=int, nargs="+", required=True)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--clean-bddl")
    parser.add_argument("--clean-language")
    parser.add_argument("--also-benchmark-variant", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import experiments.robot.libero.run_libero_eval as official_eval
    from experiments.robot.libero.run_libero_eval import (
        GenerateConfig,
        get_image_resize_size,
        initialize_model,
        run_task,
        set_seed_everywhere,
    )
    from libero.libero import benchmark

    if not args.save_video:

        def skip_video(*unused_args, **unused_kwargs):
            return None

        official_eval.save_rollout_video = skip_video
    episode_outcomes = []
    original_run_episode = official_eval.run_episode

    def record_episode(*episode_args, **episode_kwargs):
        result = original_run_episode(*episode_args, **episode_kwargs)
        episode_outcomes.append(bool(result[0]))
        return result

    official_eval.run_episode = record_episode
    cfg = GenerateConfig(
        pretrained_checkpoint=str(args.checkpoint.resolve()),
        task_suite_name=args.suite,
        num_trials_per_task=args.trials,
        num_images_in_input=2,
        use_proprio=True,
        use_l1_regression=True,
        use_diffusion=False,
        load_in_4bit=True,
        unnorm_key=f"{args.suite}_no_noops",
        use_wandb=False,
        seed=args.seed,
        local_log_dir=str(args.output.parent.resolve()),
    )
    set_seed_everywhere(args.seed)
    model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)
    resize_size = get_image_resize_size(cfg)
    base_suite = benchmark.get_benchmark_dict()[args.suite]()
    suites = [("benchmark_task", base_suite)]
    if args.clean_bddl:
        if len(args.task_ids) != 1 or not args.clean_language:
            raise ValueError("clean counterfactual requires one task ID and --clean-language")
        clean_suite = CounterfactualCleanSuite(
            base_suite, args.task_ids[0], args.clean_bddl, args.clean_language
        )
        suites = [("counterfactual_clean", clean_suite)]
        if args.also_benchmark_variant:
            shifted_suite = LanguageOverrideSuite(base_suite, args.task_ids[0], args.clean_language)
            suites.append(("benchmark_task", shifted_suite))
    results = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.with_suffix(".log").open("w") as log_file:
        for condition, task_suite in suites:
            for task_id in args.task_ids:
                episode_outcomes.clear()
                episodes, successes = run_task(
                    cfg,
                    task_suite,
                    task_id,
                    model,
                    resize_size,
                    processor,
                    action_head,
                    proprio_projector,
                    noisy_action_projector,
                    log_file=log_file,
                )
                task = task_suite.get_task(task_id)
                results.append(
                    {
                        "condition": condition,
                        "task_id": task_id,
                        "task_name": task.name,
                        "language": task.language,
                        "episodes": episodes,
                        "successes": successes,
                        "success_rate": successes / max(episodes, 1),
                        "success_by_episode": episode_outcomes.copy(),
                    }
                )
    report = {
        "schema_version": 1,
        "suite": args.suite,
        "checkpoint": str(args.checkpoint),
        "trials_per_task": args.trials,
        "seed": args.seed,
        "results": results,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
