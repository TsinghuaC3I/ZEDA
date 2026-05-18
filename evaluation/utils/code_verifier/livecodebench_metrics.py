from __future__ import annotations

import json
import multiprocessing
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm

from .pass_k_utils import compute_metrics_from_results
from .livecodebench_testing import run_test


sys.set_int_max_str_digits(50000)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _temp_run(sample, generation, debug, result, metadata_list, timeout):
    res, metadata = run_test(sample, test=generation, debug=debug, timeout=timeout)
    result.append(res)
    metadata_list.append(metadata)


def check_correctness(sample, generation, timeout, debug=True, use_process=True):
    if not use_process:
        return run_test(sample, test=generation, debug=debug, timeout=timeout)

    manager = multiprocessing.Manager()
    result = manager.list()
    metadata_list = manager.list()
    process = multiprocessing.Process(
        target=_temp_run,
        args=(sample, generation, debug, result, metadata_list, timeout),
    )
    process.start()
    process.join(
        timeout=(timeout + 1) * len(json.loads(sample["input_output"])["inputs"]) + 5
    )
    if process.is_alive():
        process.kill()
    if not result:
        in_outs = json.loads(sample["input_output"])
        result = [[-1 for _ in range(len(in_outs["inputs"]))]]
        if debug:
            print("global timeout")
        metadata_list.append(
            {"error_code": -1, "error_message": "Global Timeout", "error": "timeout"}
        )
    return result[0], metadata_list[0]


def evaluate_generations_by_problem(args):
    problem_generations = args[0]
    sample = args[1]
    debug = args[2]
    timeout = args[3]
    use_process = args[4]

    results = []
    metadata = []
    for output in problem_generations:
        current_results = [-2]
        try:
            current_results, current_metadata = check_correctness(
                sample,
                output,
                timeout=timeout,
                debug=debug,
                use_process=use_process,
            )
            fixed = []
            for entry in current_results:
                if isinstance(entry, np.ndarray):
                    entry = entry.item(0)
                if isinstance(entry, np.bool_):
                    entry = bool(entry)
                fixed.append(entry)
            current_results = fixed
        except Exception as exc:
            current_metadata = {
                "error": repr(exc),
                "error_code": -5,
                "error_message": "TestRunnerError",
            }
        finally:
            assert isinstance(current_results, list), current_results
            assert isinstance(current_metadata, dict), current_metadata
            results.append(current_results)
            metadata.append(current_metadata)
    return results, metadata


def evaluate_generations(
    samples_list: list,
    generations_list: list[list[str]],
    debug: bool = False,
    num_process_evaluate: int = 16,
    timeout=6,
):
    inputs = [
        [
            (
                generations_list[index],
                samples_list[index],
                debug,
                timeout,
                not (debug or num_process_evaluate <= 1),
            ),
            index,
        ]
        for index in range(len(generations_list))
    ]

    if debug or num_process_evaluate <= 1:
        results = {}
        metadata = {}
        with tqdm(total=len(inputs)) as progress:
            for arg, index in inputs:
                results[index], metadata[index] = evaluate_generations_by_problem(arg)
                progress.update(1)
        return results, metadata

    with tqdm(total=len(inputs)) as progress:
        with ProcessPoolExecutor(max_workers=num_process_evaluate) as executor:
            futures = {
                executor.submit(evaluate_generations_by_problem, arg): index
                for arg, index in inputs
            }

            results = {}
            metadata = {}
            for future in as_completed(futures):
                index = futures[future]
                results[index], metadata[index] = future.result()
                progress.update(1)

    assert len(results) == len(inputs), (
        f"results = {len(results)} inputs = {len(inputs)} {results=}"
    )
    return results, metadata


def livecodebench_metrics(
    samples_list,
    generations_list,
    k_list=None,
    num_process_evaluate=16,
    timeout=6,
    debug=False,
):
    if k_list is None:
        k_list = [1, 5]

    samples_linear = []
    generations_linear = []
    remap_index = []
    results = defaultdict(list)
    metadatas = defaultdict(list)

    for idx, (sample, generation_list) in enumerate(zip(samples_list, generations_list)):
        assert isinstance(generation_list, list), generations_list[0]
        for generation in generation_list:
            assert isinstance(generation, str), generations_list[0]
            samples_linear.append(sample)
            generations_linear.append([generation])
            remap_index.append(idx)

    print(f"Evaluating {len(samples_linear)}...")

    results_linear, metadatas_linear = evaluate_generations(
        samples_linear,
        generations_linear,
        debug=debug,
        num_process_evaluate=num_process_evaluate,
        timeout=timeout,
    )

    for idx, sub_results in sorted(results_linear.items(), key=lambda item: item[0]):
        results[remap_index[idx]].append(sub_results[0])

    for idx, sub_metadata in sorted(metadatas_linear.items(), key=lambda item: item[0]):
        metadatas[remap_index[idx]].append(sub_metadata[0])

    metrics = compute_metrics_from_results(results, k_list=k_list)

    final_metadata = []
    for key in sorted(metadatas.keys()):
        final_metadata.append(metadatas[key])
    for idx, metadata_list in enumerate(final_metadata):
        if type(metadata_list) is not list:
            final_metadata[idx] = [json.dumps(metadata_list)]
        else:
            final_metadata[idx] = [json.dumps(item) for item in metadata_list]
        assert len(final_metadata[idx]) == len(generations_list[0])

    return [metrics, results, final_metadata]


# Backward-compatible alias while imports migrate.
codegen_metrics = livecodebench_metrics
