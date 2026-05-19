#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import pickle
import statistics
import zlib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
VALIDATION_SCHEMA_VERSION = 2
SUMMARY_FIELDNAMES = [
    "raw_output_file",
    "benchmark_file",
    "domain",
    "n",
    "total",
    "correct",
    "accuracy",
    "pass_at_n",
    "mean_at_n",
    "strict_prompt_accuracy",
    "strict_instruction_accuracy",
    "loose_prompt_accuracy",
    "loose_instruction_accuracy",
    "missing_uid",
    "missing_answer",
    "completion_tokens",
    "zero_expert_ratio",
    "zero_expert_ratio_std",
    "e2e_latency",
]

# The IF verifiers rely on vendored NLTK assets under evaluation/benchmark/nltk_data.
os.environ["NLTK_DATA"] = str(SCRIPT_DIR / "benchmark" / "nltk_data")


def load_benchmark_rows(benchmark_file: str) -> Dict[str, Dict[str, Any]]:
    benchmark_rows = {}
    with open(benchmark_file, "r", encoding="utf-8") as file_obj:
        for line_number, line in enumerate(file_obj, 1):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            uid = data.get("uid")
            if not uid:
                raise ValueError(f"missing uid in {benchmark_file}:{line_number}")
            benchmark_rows[uid] = data
    return benchmark_rows


def load_model_responses(raw_output_file: str) -> Dict[str, List[Dict[str, Any]]]:
    responses: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    with open(raw_output_file, "r", encoding="utf-8") as file_obj:
        for line_number, line in enumerate(file_obj, 1):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            uid = data.get("uid")
            if not uid:
                raise ValueError(f"missing uid in {raw_output_file}:{line_number}")
            sample_index = data.get("sample_index", 0)
            if not isinstance(sample_index, int):
                raise ValueError(
                    f"invalid sample_index in {raw_output_file}:{line_number}: "
                    f"{sample_index!r}"
                )
            data["sample_index"] = sample_index
            responses[uid].append(data)
    return responses


def load_manifest(raw_output_file: str) -> Optional[Dict[str, Any]]:
    manifest_path = Path(f"{raw_output_file}.meta.json")
    if not manifest_path.exists():
        return None
    with manifest_path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def find_matching_benchmark(raw_output_file: Path, benchmark_files: List[Path]) -> Path:
    manifest = load_manifest(str(raw_output_file))
    if manifest and manifest.get("input_path"):
        return Path(manifest["input_path"])

    for benchmark_file in benchmark_files:
        if benchmark_file.stem in raw_output_file.stem:
            return benchmark_file
    raise ValueError(f"unable to find benchmark file for {raw_output_file}")


def resolve_effective_n(args_n: Optional[int], raw_output_file: str) -> int:
    candidate = args_n
    if candidate is None:
        manifest = load_manifest(raw_output_file)
        if manifest is not None:
            candidate = manifest.get("n")

    if not isinstance(candidate, int) or candidate <= 0:
        raise ValueError(
            f"evaluation requires --n or a manifest with a positive integer n for "
            f"{raw_output_file}"
        )
    return candidate


def compute_stats(raw_output_file: str) -> Dict[str, Any]:
    stats: Dict[str, List[float]] = {}
    register_keys = ["completion_tokens", "normal_expert_ratio", "e2e_latency"]
    with open(raw_output_file, "r", encoding="utf-8") as file_obj:
        for line in file_obj:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            meta_info = data.get("meta", {})
            if "completion_tokens" in meta_info:
                stats.setdefault("completion_tokens", []).append(
                    meta_info["completion_tokens"]
                )
            for key in register_keys[1:]:
                if key in meta_info:
                    stats.setdefault(key, []).append(meta_info[key])

    normal_expert_ratios = list(stats.get("normal_expert_ratio", []))
    aggregated: Dict[str, Any] = {}
    for key, values in stats.items():
        aggregated[key] = sum(values) / len(values)

    if normal_expert_ratios:
        zce_ratios = [1 - value for value in normal_expert_ratios]
        aggregated["zero_expert_ratio_std"] = statistics.pstdev(zce_ratios)
    else:
        aggregated["zero_expert_ratio_std"] = None
    return aggregated


@dataclass(frozen=True)
class BenchmarkSpec:
    domain: str
    benchmark_type: Optional[str] = None


def get_benchmark_spec(benchmark_file: Path) -> BenchmarkSpec:
    benchmark_name = benchmark_file.stem.lower()
    if "livecodebench" in benchmark_name:
        return BenchmarkSpec(domain="code", benchmark_type="livecodebench")
    if "humanevalplus" in benchmark_name or "mbppplus" in benchmark_name:
        return BenchmarkSpec(domain="code", benchmark_type="evalplus")
    if benchmark_name in {"gsm8k", "math-500"} or benchmark_name.startswith("aime-") or benchmark_name.startswith("mmlu") or benchmark_name.startswith("gpqa"):
        return BenchmarkSpec(domain="math")
    if benchmark_name == "ifeval":
        return BenchmarkSpec(domain="if", benchmark_type="ifeval")
    if benchmark_name == "ifbench":
        return BenchmarkSpec(domain="if", benchmark_type="ifbench")
    raise ValueError(f"unsupported benchmark file: {benchmark_file.name}")


def maybe_int(value: float | int) -> float | int:
    if isinstance(value, int):
        return value
    if float(value).is_integer():
        return int(value)
    return value


def format_optional_percentage(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{value:.2%}"


def normalize_if_response(response_data: Optional[Dict[str, Any]]) -> str:
    if response_data is None:
        return ""

    raw_response = response_data.get("response")
    if not isinstance(raw_response, str):
        return ""

    response_text = raw_response.strip()
    if "</think>" in response_text:
        response_text = response_text.split("</think>")[-1].strip()
    if "<|im_end|>" in response_text:
        response_text = response_text.split("<|im_end|>")[0].strip()
    return response_text


def group_responses_by_sample(
    benchmark_rows: Dict[str, Dict[str, Any]],
    responses_by_uid: Dict[str, List[Dict[str, Any]]],
    n: int,
    *,
    require_complete: bool,
) -> Dict[str, Dict[int, Dict[str, Any]]]:
    grouped: Dict[str, Dict[int, Dict[str, Any]]] = {}
    missing: Dict[str, List[int]] = {}
    duplicates: Dict[str, List[int]] = {}

    for uid in benchmark_rows:
        index_map: Dict[int, Dict[str, Any]] = {}
        for response in responses_by_uid.get(uid, []):
            sample_index = response["sample_index"]
            if sample_index in index_map:
                duplicates.setdefault(uid, []).append(sample_index)
                continue
            if sample_index < 0 or sample_index >= n:
                raise ValueError(
                    f"{uid} has unexpected sample indices outside 0..{n-1}: "
                    f"{[sample_index]}"
                )
            index_map[sample_index] = response

        if require_complete:
            expected = set(range(n))
            missing_indices = sorted(expected - set(index_map.keys()))
            if missing_indices:
                missing[uid] = missing_indices
        grouped[uid] = index_map

    if duplicates:
        raise ValueError(f"duplicate sample_index entries detected: {duplicates}")
    if missing:
        raise ValueError(f"incomplete sampled rollout detected: {missing}")
    return grouped


def mean_boolean(values: List[bool]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def normalize_math_answer(answer: Any, extract_answer_fn) -> str:
    normalized = str(answer)
    if "\\boxed" in normalized:
        extracted = extract_answer_fn(normalized)
        if extracted is not None:
            return extracted
    return normalized


def compute_math_reward(
    benchmark_rows: Dict[str, Dict[str, Any]],
    responses_by_uid: Dict[str, List[Dict[str, Any]]],
    n: int,
) -> Dict[str, Any]:
    from utils.math_verifier import (
        extract_answer,
        grade_answer_mathd,
        grade_answer_sympy,
    )

    grouped = group_responses_by_sample(
        benchmark_rows,
        responses_by_uid,
        n,
        require_complete=False,
    )

    total = len(benchmark_rows)
    correct = 0
    pass_at_n_correct = 0
    total_slot_correct = 0
    missing_uid = 0
    missing_answer = 0
    missing_response_slots = 0
    missing_answer_slots = 0
    results = []

    for uid, benchmark_row in benchmark_rows.items():
        correct_answer = normalize_math_answer(
            benchmark_row.get("label", ""),
            extract_answer,
        )
        sample_records = []
        sample_flags: List[bool] = []

        for sample_index in range(n):
            response_data = grouped[uid].get(sample_index)
            if response_data is None:
                missing_response_slots += 1
                if sample_index == 0:
                    missing_uid += 1
                sample_records.append(
                    {
                        "sample_index": sample_index,
                        "correct": False,
                        "model_answer": None,
                        "error": "missing_response",
                    }
                )
                sample_flags.append(False)
                continue

            model_answer = extract_answer(response_data.get("response", ""))
            if model_answer is None:
                missing_answer_slots += 1
                if sample_index == 0:
                    missing_answer += 1
                sample_records.append(
                    {
                        "sample_index": sample_index,
                        "correct": False,
                        "model_answer": None,
                        "error": "missing_answer",
                    }
                )
                sample_flags.append(False)
                continue

            is_correct = grade_answer_mathd(
                model_answer,
                correct_answer,
            ) or grade_answer_sympy(model_answer, correct_answer)
            sample_records.append(
                {
                    "sample_index": sample_index,
                    "correct": is_correct,
                    "model_answer": model_answer,
                    "error": None,
                }
            )
            sample_flags.append(is_correct)

        if sample_flags[0]:
            correct += 1
        if any(sample_flags):
            pass_at_n_correct += 1
        total_slot_correct += sum(sample_flags)

        results.append(
            {
                "uid": uid,
                "correct_answer": correct_answer,
                "correct": sample_flags[0],
                "pass_at_n": any(sample_flags),
                "mean_at_n": mean_boolean(sample_flags),
                "samples": sample_records,
            }
        )

    mean_at_n = total_slot_correct / (total * n) if total > 0 else 0.0
    pass_at_n = pass_at_n_correct / total if total > 0 else 0.0
    return {
        "domain": "math",
        "n": n,
        "total": total,
        "correct": correct,
        "accuracy": mean_at_n,
        "pass_at_n": pass_at_n,
        "mean_at_n": mean_at_n,
        "pass_at_n_correct": pass_at_n_correct,
        "missing_uid": missing_uid,
        "missing_answer": missing_answer,
        "missing_response_slots": missing_response_slots,
        "missing_answer_slots": missing_answer_slots,
        "results": results,
    }


def build_code_sample(benchmark_row: Dict[str, Any]) -> Dict[str, str]:
    all_tests = benchmark_row.get("public_test_cases", []) + decode_private_test_cases(
        benchmark_row.get("private_test_cases", [])
    )
    return {
        "input_output": json.dumps(
            {
                "inputs": [test_case["input"] for test_case in all_tests],
                "outputs": [test_case["output"] for test_case in all_tests],
                "fn_name": benchmark_row.get("metadata", {}).get("func_name"),
            }
        )
    }


def decode_private_test_cases(raw_value: Any) -> List[Dict[str, Any]]:
    if isinstance(raw_value, list):
        return raw_value
    if isinstance(raw_value, dict):
        return [raw_value]
    if not raw_value:
        return []

    try:
        decoded = json.loads(raw_value)
    except Exception:
        unpacked = pickle.loads(
            zlib.decompress(base64.b64decode(raw_value.encode("utf-8")))
        )
        decoded = json.loads(unpacked)

    if isinstance(decoded, list):
        return decoded
    raise ValueError("decoded private_test_cases is not a list")


def compute_code_reward(
    benchmark_type: str,
    benchmark_rows: Dict[str, Dict[str, Any]],
    responses_by_uid: Dict[str, List[Dict[str, Any]]],
    n: int,
    num_process_evaluate: int,
    timeout: int,
    debug: bool,
) -> Dict[str, Any]:
    if benchmark_type == "evalplus":
        return compute_evalplus_reward(
            benchmark_rows=benchmark_rows,
            responses_by_uid=responses_by_uid,
            n=n,
            num_process_evaluate=num_process_evaluate,
            timeout=timeout,
        )
    if benchmark_type == "livecodebench":
        return compute_livecodebench_reward(
            benchmark_rows=benchmark_rows,
            responses_by_uid=responses_by_uid,
            n=n,
            num_process_evaluate=num_process_evaluate,
            timeout=timeout,
            debug=debug,
        )
    raise ValueError(f"unsupported code benchmark type: {benchmark_type}")


def validate_shared_if_totals(sample_summaries: List[Dict[str, Any]], key: str) -> int:
    totals = [summary[key] for summary in sample_summaries]
    first_total = totals[0]
    if any(total != first_total for total in totals[1:]):
        raise ValueError(f"inconsistent IF totals for {key}: {totals}")
    return first_total


def aggregate_if_breakdown(sample_breakdowns: List[Dict[str, Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    all_keys = sorted(
        {
            instruction_id
            for breakdown in sample_breakdowns
            for instruction_id in breakdown.keys()
        }
    )
    aggregated: Dict[str, Dict[str, Any]] = {}
    for instruction_id in all_keys:
        totals = [breakdown[instruction_id]["total"] for breakdown in sample_breakdowns]
        total = totals[0]
        if any(value != total for value in totals[1:]):
            raise ValueError(
                f"inconsistent IF totals for {instruction_id}: {totals}"
            )
        correct = sum(
            breakdown[instruction_id]["correct"] for breakdown in sample_breakdowns
        ) / len(sample_breakdowns)
        aggregated[instruction_id] = {
            "correct": maybe_int(correct),
            "total": total,
            "accuracy": correct / total if total else 0.0,
        }
    return aggregated


def aggregate_if_summary(sample_summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not sample_summaries:
        return {
            "prompt_total": 0,
            "prompt_correct": 0,
            "prompt_accuracy": 0.0,
            "instruction_total": 0,
            "instruction_correct": 0,
            "instruction_accuracy": 0.0,
            "tier0": {},
            "tier1": {},
        }

    prompt_total = validate_shared_if_totals(sample_summaries, "prompt_total")
    instruction_total = validate_shared_if_totals(sample_summaries, "instruction_total")
    prompt_correct = sum(
        summary["prompt_correct"] for summary in sample_summaries
    ) / len(sample_summaries)
    instruction_correct = sum(
        summary["instruction_correct"] for summary in sample_summaries
    ) / len(sample_summaries)

    return {
        "prompt_total": prompt_total,
        "prompt_correct": maybe_int(prompt_correct),
        "prompt_accuracy": prompt_correct / prompt_total if prompt_total else 0.0,
        "instruction_total": instruction_total,
        "instruction_correct": maybe_int(instruction_correct),
        "instruction_accuracy": (
            instruction_correct / instruction_total if instruction_total else 0.0
        ),
        "tier0": aggregate_if_breakdown(
            [summary["tier0"] for summary in sample_summaries]
        ),
        "tier1": aggregate_if_breakdown(
            [summary["tier1"] for summary in sample_summaries]
        ),
    }


def compute_if_reward(
    benchmark_type: str,
    benchmark_rows: Dict[str, Dict[str, Any]],
    responses_by_uid: Dict[str, List[Dict[str, Any]]],
    n: int,
) -> Dict[str, Any]:
    if benchmark_type == "ifeval":
        from utils.if_verifier.ifeval import evaluate_ifeval_responses

        evaluator = evaluate_ifeval_responses
        score_variant = "strict"
    elif benchmark_type == "ifbench":
        from utils.if_verifier.ifbench import evaluate_ifbench_responses

        evaluator = evaluate_ifbench_responses
        score_variant = "loose"
    else:
        raise ValueError(f"unsupported if benchmark type: {benchmark_type}")

    grouped = group_responses_by_sample(
        benchmark_rows,
        responses_by_uid,
        n,
        require_complete=False,
    )

    total = len(benchmark_rows)
    missing_uid = 0
    missing_answer = 0
    missing_response_slots = 0
    missing_answer_slots = 0
    sample_strict_summaries = []
    sample_loose_summaries = []
    per_uid_samples: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for sample_index in range(n):
        examples = []
        sample_state: Dict[str, Dict[str, Any]] = {}
        for uid, benchmark_row in benchmark_rows.items():
            response_data = grouped[uid].get(sample_index)
            response_text = ""
            error = None
            if response_data is None:
                missing_response_slots += 1
                if sample_index == 0:
                    missing_uid += 1
                error = "missing_response"
            else:
                response_text = normalize_if_response(response_data)
                if not response_text:
                    missing_answer_slots += 1
                    if sample_index == 0:
                        missing_answer += 1
                    error = "missing_answer"

            sample_state[uid] = {
                "sample_index": sample_index,
                "response": response_text,
                "error": error,
            }
            examples.append(
                {
                    "uid": uid,
                    "key": benchmark_row["key"],
                    "instruction_id_list": benchmark_row["instruction_id_list"],
                    "prompt": benchmark_row["prompt"][0]["content"],
                    "kwargs": benchmark_row["kwargs"],
                    "response": response_text,
                }
            )

        evaluation = evaluator(examples)
        if len(evaluation["results"]) != total:
            raise ValueError(
                f"IF evaluation returned {len(evaluation['results'])} results for "
                f"{total} prompts at sample_index={sample_index}"
            )

        sample_strict_summaries.append(evaluation["strict"])
        sample_loose_summaries.append(evaluation["loose"])

        for result in evaluation["results"]:
            uid = result["uid"]
            per_uid_samples[uid].append(
                {
                    **sample_state[uid],
                    "strict": result["strict"],
                    "loose": result["loose"],
                }
            )

    strict_stats = aggregate_if_summary(sample_strict_summaries)
    loose_stats = aggregate_if_summary(sample_loose_summaries)
    score_stats = loose_stats if score_variant == "loose" else strict_stats

    results = []
    for uid, benchmark_row in benchmark_rows.items():
        ordered_samples = sorted(
            per_uid_samples[uid],
            key=lambda item: item["sample_index"],
        )
        results.append(
            {
                "uid": uid,
                "key": benchmark_row["key"],
                "instruction_id_list": benchmark_row["instruction_id_list"],
                "samples": ordered_samples,
            }
        )

    return {
        "domain": "if",
        "if_benchmark_type": benchmark_type,
        "if_score_variant": score_variant,
        "n": n,
        "total": total,
        "correct": score_stats["prompt_correct"],
        "accuracy": score_stats["prompt_accuracy"],
        "missing_uid": missing_uid,
        "missing_answer": missing_answer,
        "missing_response_slots": missing_response_slots,
        "missing_answer_slots": missing_answer_slots,
        "strict": strict_stats,
        "loose": loose_stats,
        "strict_prompt_accuracy": strict_stats["prompt_accuracy"],
        "strict_instruction_accuracy": strict_stats["instruction_accuracy"],
        "loose_prompt_accuracy": loose_stats["prompt_accuracy"],
        "loose_instruction_accuracy": loose_stats["instruction_accuracy"],
        "results": results,
    }


def compute_livecodebench_reward(
    benchmark_rows: Dict[str, Dict[str, Any]],
    responses_by_uid: Dict[str, List[Dict[str, Any]]],
    n: int,
    num_process_evaluate: int,
    timeout: int,
    debug: bool,
) -> Dict[str, Any]:
    from utils.code_verifier import livecodebench_metrics
    from utils.code_verifier.extraction import extract_code

    grouped = group_responses_by_sample(
        benchmark_rows,
        responses_by_uid,
        n,
        require_complete=True,
    )
    sorted_benchmark_rows = sorted(
        benchmark_rows.values(), key=lambda row: row["question_id"]
    )

    samples_list = []
    generations_list = []
    for benchmark_row in sorted_benchmark_rows:
        uid = benchmark_row["uid"]
        sample_map = grouped[uid]
        samples_list.append(build_code_sample(benchmark_row))
        generations_list.append(
            [
                extract_code(sample_map[sample_index].get("response"))
                for sample_index in range(n)
            ]
        )

    k_list = sorted({1, n})
    metrics, raw_results, metadata = livecodebench_metrics(
        samples_list=samples_list,
        generations_list=generations_list,
        k_list=k_list,
        num_process_evaluate=num_process_evaluate,
        timeout=timeout,
        debug=debug,
    )

    per_problem_results = []
    graded_lists = []
    for idx, benchmark_row in enumerate(sorted_benchmark_rows):
        task_results = raw_results[idx]
        graded_list = [
            all(test_result > 0 for test_result in sample_result)
            for sample_result in task_results
        ]
        graded_lists.append(graded_list)
        per_problem_results.append(
            {
                "uid": benchmark_row["uid"],
                "question_id": benchmark_row["question_id"],
                "graded_list": graded_list,
                "pass_rate": sum(graded_list) / len(graded_list),
                "metadata": metadata[idx],
            }
        )

    mean_at_n = (
        sum(sum(graded_list) / len(graded_list) for graded_list in graded_lists)
        / len(graded_lists)
        if graded_lists
        else 0.0
    )
    pass_at_n = metrics.get(f"pass@{n}")
    if pass_at_n is not None:
        pass_at_n = float(pass_at_n)

    return {
        "domain": "code",
        "code_benchmark_type": "livecodebench",
        "n": n,
        "total": len(sorted_benchmark_rows),
        "pass_at_n": pass_at_n,
        "mean_at_n": mean_at_n,
        "results": per_problem_results,
        "detail": metrics.get("detail", {}),
    }


def compute_evalplus_reward(
    benchmark_rows: Dict[str, Dict[str, Any]],
    responses_by_uid: Dict[str, List[Dict[str, Any]]],
    n: int,
    num_process_evaluate: int,
    timeout: int,
) -> Dict[str, Any]:
    from utils.code_verifier import evalplus_metrics
    from utils.code_verifier.extraction import extract_code

    grouped = group_responses_by_sample(
        benchmark_rows,
        responses_by_uid,
        n,
        require_complete=True,
    )
    sorted_benchmark_rows = sorted(
        benchmark_rows.values(),
        key=lambda row: str(row["task_id"]),
    )

    problems_list = []
    generations_list = []
    for benchmark_row in sorted_benchmark_rows:
        uid = benchmark_row["uid"]
        sample_map = grouped[uid]
        problems_list.append(benchmark_row)
        generations_list.append(
            [
                extract_code(
                    sample_map[sample_index].get("response"),
                    preserve_indentation=True,
                )
                for sample_index in range(n)
            ]
        )

    k_list = sorted({1, n})
    metrics, raw_results, metadata = evalplus_metrics(
        problems_list=problems_list,
        generations_list=generations_list,
        k_list=k_list,
        num_process_evaluate=num_process_evaluate,
        timeout=timeout,
    )

    per_problem_results = []
    graded_lists = []
    for idx, benchmark_row in enumerate(sorted_benchmark_rows):
        task_results = raw_results[idx]
        graded_list = [all(sample_result) for sample_result in task_results]
        graded_lists.append(graded_list)
        per_problem_results.append(
            {
                "uid": benchmark_row["uid"],
                "task_id": benchmark_row["task_id"],
                "graded_list": graded_list,
                "pass_rate": sum(graded_list) / len(graded_list),
                "metadata": metadata[idx],
            }
        )

    mean_at_n = (
        sum(sum(graded_list) / len(graded_list) for graded_list in graded_lists)
        / len(graded_lists)
        if graded_lists
        else 0.0
    )
    pass_at_n = metrics.get(f"pass@{n}")
    if pass_at_n is not None:
        pass_at_n = float(pass_at_n)

    return {
        "domain": "code",
        "code_benchmark_type": "evalplus",
        "n": n,
        "total": len(sorted_benchmark_rows),
        "pass_at_n": pass_at_n,
        "mean_at_n": mean_at_n,
        "results": per_problem_results,
        "detail": metrics.get("detail", {}),
    }


def evaluate_file(
    benchmark_file: Path,
    raw_output_file: Path,
    args: argparse.Namespace,
    n: int,
) -> Dict[str, Any]:
    benchmark_rows = load_benchmark_rows(str(benchmark_file))
    responses_by_uid = load_model_responses(str(raw_output_file))
    benchmark_spec = get_benchmark_spec(benchmark_file)

    if benchmark_spec.domain == "code":
        stats = compute_code_reward(
            benchmark_type=benchmark_spec.benchmark_type or "",
            benchmark_rows=benchmark_rows,
            responses_by_uid=responses_by_uid,
            n=n,
            num_process_evaluate=args.num_process_evaluate,
            timeout=args.timeout,
            debug=args.debug,
        )
    elif benchmark_spec.domain == "math":
        stats = compute_math_reward(
            benchmark_rows=benchmark_rows,
            responses_by_uid=responses_by_uid,
            n=n,
        )
    elif benchmark_spec.domain == "if":
        stats = compute_if_reward(
            benchmark_type=benchmark_spec.benchmark_type or "",
            benchmark_rows=benchmark_rows,
            responses_by_uid=responses_by_uid,
            n=n,
        )
    else:
        raise ValueError(f"unsupported benchmark domain: {benchmark_spec.domain}")

    stats.update(compute_stats(str(raw_output_file)))
    stats["schema_version"] = VALIDATION_SCHEMA_VERSION
    return stats


def get_validation_file(raw_output_file: Path) -> Path:
    return raw_output_file.parent / f"{raw_output_file.stem}_validation.json"


def load_cached_validation(validation_file: Path, expected_n: int) -> Optional[Dict[str, Any]]:
    if not validation_file.exists():
        return None

    with validation_file.open("r", encoding="utf-8") as file_obj:
        stats = json.load(file_obj)

    if not isinstance(stats, dict):
        raise ValueError(f"invalid cached validation file: {validation_file}")
    if stats.get("n") != expected_n:
        raise ValueError(
            f"cached validation file {validation_file} was computed with n="
            f"{stats.get('n')!r}, expected {expected_n}"
        )
    return stats


def build_summary_row(
    benchmark_file: Path,
    raw_output_file: Path,
    stats: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "raw_output_file": raw_output_file.name,
        "benchmark_file": benchmark_file.name,
        "domain": stats.get("domain"),
        "n": stats.get("n"),
        "total": stats.get("total", 0),
        "correct": stats.get("correct"),
        "accuracy": stats.get("accuracy"),
        "pass_at_n": stats.get("pass_at_n"),
        "mean_at_n": stats.get("mean_at_n"),
        "strict_prompt_accuracy": stats.get("strict_prompt_accuracy"),
        "strict_instruction_accuracy": stats.get("strict_instruction_accuracy"),
        "loose_prompt_accuracy": stats.get("loose_prompt_accuracy"),
        "loose_instruction_accuracy": stats.get("loose_instruction_accuracy"),
        "missing_uid": stats.get("missing_uid"),
        "missing_answer": stats.get("missing_answer"),
        "completion_tokens": stats.get("completion_tokens"),
        "zero_expert_ratio": (
            1 - stats["normal_expert_ratio"]
            if stats.get("normal_expert_ratio") is not None
            else None
        ),
        "zero_expert_ratio_std": stats.get("zero_expert_ratio_std"),
        "e2e_latency": stats.get("e2e_latency"),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Offline benchmark verification")
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--num-process-evaluate", type=int, default=12)
    parser.add_argument("--timeout", type=int, default=6)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--raw-output-dir", default="raw_output/")
    parser.add_argument("--pattern", default="*.jsonl")
    return parser.parse_args()


def main():
    args = parse_args()
    benchmark_dir = SCRIPT_DIR / "benchmark"
    raw_output_dir = Path(args.raw_output_dir)
    if not raw_output_dir.is_absolute():
        raw_output_dir = SCRIPT_DIR / raw_output_dir

    benchmark_files = sorted(benchmark_dir.glob("*.jsonl"))
    if not benchmark_files:
        print(f"错误：在 {benchmark_dir} 中找不到benchmark文件")
        return

    raw_output_files = sorted(raw_output_dir.glob(args.pattern))
    if not raw_output_files:
        print(f"错误：在 {raw_output_dir} 中找不到模型回复文件")
        return

    summary_rows = []
    for raw_output_file in raw_output_files:
        try:
            print(f"\n{'=' * 60}")
            print(f"处理文件: {raw_output_file.name}")
            print(f"{'=' * 60}")

            benchmark_file = find_matching_benchmark(raw_output_file, benchmark_files)
            effective_n = resolve_effective_n(args.n, str(raw_output_file))
            output_file = get_validation_file(raw_output_file)
            stats = None

            if output_file.exists():
                stats = load_cached_validation(output_file, expected_n=effective_n)
                print(f"检测到已有验证结果，跳过 verify: {output_file.name}")

            if stats is None:
                print("未找到可用的验证缓存，开始 verify...")
                stats = evaluate_file(benchmark_file, raw_output_file, args, n=effective_n)
                with output_file.open("w", encoding="utf-8") as file_obj:
                    json.dump(stats, file_obj, ensure_ascii=False, indent=2)
                print(f"\n详细结果已保存到: {output_file}")

            print("\n统计结果:")
            print(f"  domain: {stats['domain']}")
            print(f"  总题目数: {stats['total']}")
            if stats["domain"] == "code":
                print(f"  n: {stats['n']}")
                print(f"  pass@n: {stats.get('pass_at_n')}")
                print(f"  mean@n: {stats.get('mean_at_n')}")
            elif stats["domain"] == "if":
                print(f"  n: {stats['n']}")
                print(f"  accuracy: {format_optional_percentage(stats.get('accuracy'))}")
                print(
                    f"  strict prompt accuracy: "
                    f"{format_optional_percentage(stats.get('strict_prompt_accuracy'))}"
                )
                print(
                    f"  strict instruction accuracy: "
                    f"{format_optional_percentage(stats.get('strict_instruction_accuracy'))}"
                )
                print(
                    f"  loose prompt accuracy: "
                    f"{format_optional_percentage(stats.get('loose_prompt_accuracy'))}"
                )
                print(
                    f"  loose instruction accuracy: "
                    f"{format_optional_percentage(stats.get('loose_instruction_accuracy'))}"
                )
                print(f"  缺失UID: {stats['missing_uid']}")
                print(f"  缺失答案: {stats['missing_answer']}")
            else:
                print(f"  n: {stats['n']}")
                print(f"  正确答案数: {stats['correct']}")
                print(f"  准确率: {format_optional_percentage(stats.get('accuracy'))}")
                print(f"  pass@n: {format_optional_percentage(stats.get('pass_at_n'))}")
                print(f"  mean@n: {format_optional_percentage(stats.get('mean_at_n'))}")
                print(f"  缺失UID: {stats['missing_uid']}")
                print(f"  缺失答案: {stats['missing_answer']}")

            summary_rows.append(build_summary_row(benchmark_file, raw_output_file, stats))
        except Exception as exc:
            print(f"处理文件: {raw_output_file.name} 时发生错误: {exc}")
            break

    if summary_rows:
        csv_file = SCRIPT_DIR / "summary.csv"
        with csv_file.open("w", encoding="utf-8", newline="") as file_obj:
            writer = csv.DictWriter(file_obj, fieldnames=SUMMARY_FIELDNAMES)
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"\n汇总统计结果已保存到: {csv_file}")


if __name__ == "__main__":
    main()
