from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
from typing import Any, Dict, Iterable, Sequence


def _load_evaluation_lib():
    try:
        from . import evaluation_lib
    except ImportError as exc:
        raise RuntimeError(
            "IFBench verification requires the vendored dependencies "
            "`nltk`, `emoji`, and `syllapy`."
        ) from exc
    return evaluation_lib


def _summarize_outputs(outputs: Sequence[Any]) -> Dict[str, Any]:
    prompt_total = 0
    prompt_correct = 0
    instruction_total = 0
    instruction_correct = 0
    tier0_total: Dict[str, int] = defaultdict(int)
    tier0_correct: Dict[str, int] = defaultdict(int)
    tier1_total: Dict[str, int] = defaultdict(int)
    tier1_correct: Dict[str, int] = defaultdict(int)

    for example in outputs:
        follow_instruction_list = example.follow_instruction_list
        instruction_id_list = example.instruction_id_list

        prompt_total += 1
        if all(follow_instruction_list):
            prompt_correct += 1

        instruction_total += len(instruction_id_list)
        instruction_correct += sum(follow_instruction_list)

        for instruction_id, followed_or_not in zip(
            instruction_id_list, follow_instruction_list
        ):
            tier0_instruction_id = instruction_id.split(":")[0]
            tier0_total[tier0_instruction_id] += 1
            if followed_or_not:
                tier0_correct[tier0_instruction_id] += 1

            tier1_total[instruction_id] += 1
            if followed_or_not:
                tier1_correct[instruction_id] += 1

    def build_breakdown(
        totals: Dict[str, int], correct: Dict[str, int]
    ) -> Dict[str, Dict[str, float | int]]:
        return {
            instruction_id: {
                "correct": correct[instruction_id],
                "total": totals[instruction_id],
                "accuracy": correct[instruction_id] / totals[instruction_id],
            }
            for instruction_id in sorted(totals.keys())
        }

    return {
        "prompt_total": prompt_total,
        "prompt_correct": prompt_correct,
        "prompt_accuracy": prompt_correct / prompt_total if prompt_total else 0.0,
        "instruction_total": instruction_total,
        "instruction_correct": instruction_correct,
        "instruction_accuracy": (
            instruction_correct / instruction_total if instruction_total else 0.0
        ),
        "tier0": build_breakdown(tier0_total, tier0_correct),
        "tier1": build_breakdown(tier1_total, tier1_correct),
    }


def _serialize_output(output: Any) -> Dict[str, Any]:
    return asdict(output)


def evaluate_ifbench_responses(
    examples: Iterable[Dict[str, Any]],
) -> Dict[str, Any]:
    evaluation_lib = _load_evaluation_lib()

    strict_outputs = []
    loose_outputs = []
    results = []

    try:
        for example in examples:
            input_example = evaluation_lib.InputExample(
                key=example["key"],
                instruction_id_list=example["instruction_id_list"],
                prompt=example["prompt"],
                kwargs=example["kwargs"],
            )
            prompt_to_response = {input_example.prompt: example.get("response", "")}
            strict_output = evaluation_lib.test_instruction_following_strict(
                input_example, prompt_to_response
            )
            loose_output = evaluation_lib.test_instruction_following_loose(
                input_example, prompt_to_response
            )
            strict_outputs.append(strict_output)
            loose_outputs.append(loose_output)
            results.append(
                {
                    "uid": example["uid"],
                    "key": input_example.key,
                    "instruction_id_list": input_example.instruction_id_list,
                    "strict": _serialize_output(strict_output),
                    "loose": _serialize_output(loose_output),
                }
            )
    except Exception as exc:
        print(type(exc))
        print(repr(exc))

    return {
        "strict": _summarize_outputs(strict_outputs),
        "loose": _summarize_outputs(loose_outputs),
        "results": results,
    }
