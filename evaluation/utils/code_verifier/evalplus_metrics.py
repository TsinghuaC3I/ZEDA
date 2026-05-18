from __future__ import annotations

import ast
import contextlib
import io
import math
import multiprocessing
import os
import platform
import queue
import signal
import sys
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from multiprocessing import Array, Value
from typing import Any, Optional

import numpy as np
from tqdm import tqdm

from .extraction import extract_code
from .pass_k_utils import estimate_pass_at_k


if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(50000)


DEFAULT_GT_TIME_LIMIT_FACTOR = 4.0
DEFAULT_MIN_TIME_LIMIT = 4.0

PASS = "pass"
FAIL = "fail"
TIMEOUT = "timeout"

_SUCCESS = 0
_FAILED = 1
_TIMEOUT = 2
_UNKNOWN = 3
_STATUS_MAPPING = {_SUCCESS: PASS, _FAILED: FAIL, _TIMEOUT: TIMEOUT, _UNKNOWN: None}

MBPP_OUTPUT_NOT_NONE_TASKS = ["check_str", "text_match_three", "text_starta_endb"]
MBPP_OUTPUT_SET_EQ_TASKS = [
    "similar_elements",
    "find_char_long",
    "common_in_nested_lists",
    "extract_singly",
    "larg_nnum",
    "intersection_array",
    "find_dissimilar",
    "Diff",
]


class TimeoutException(Exception):
    pass


class WriteOnlyStringIO(io.StringIO):
    def read(self, *args, **kwargs):
        raise IOError

    def readline(self, *args, **kwargs):
        raise IOError

    def readlines(self, *args, **kwargs):
        raise IOError

    def readable(self, *args, **kwargs):
        return False


class redirect_stdin(contextlib._RedirectStream):  # type: ignore[attr-defined]
    _stream = "stdin"


@contextlib.contextmanager
def swallow_io():
    stream = WriteOnlyStringIO()
    with contextlib.redirect_stdout(stream):
        with contextlib.redirect_stderr(stream):
            with redirect_stdin(stream):
                yield


@contextlib.contextmanager
def time_limit(seconds: float):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


@contextlib.contextmanager
def chdir(root: str):
    if root == ".":
        yield
        return

    cwd = os.getcwd()
    os.chdir(root)
    try:
        yield
    finally:
        os.chdir(cwd)


@contextlib.contextmanager
def create_tempdir():
    with tempfile.TemporaryDirectory() as dirname:
        with chdir(dirname):
            yield dirname


def _query_maximum_memory_bytes() -> Optional[int]:
    maximum_memory_bytes = int(
        os.getenv("EVALPLUS_MAX_MEMORY_BYTES", 4 * 1024 * 1024 * 1024)
    )
    if maximum_memory_bytes == -1:
        return None

    try:
        total_memory = int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, OSError, ValueError):
        total_memory = None

    if total_memory is not None:
        return min(maximum_memory_bytes, total_memory)
    return maximum_memory_bytes


def reliability_guard(maximum_memory_bytes: Optional[int] = None):
    if maximum_memory_bytes is not None and platform.uname().system != "Windows":
        import resource

        resource.setrlimit(
            resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes)
        )
        resource.setrlimit(
            resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes)
        )
        if platform.uname().system != "Darwin":
            resource.setrlimit(
                resource.RLIMIT_STACK, (maximum_memory_bytes, maximum_memory_bytes)
            )

    import builtins
    import faulthandler
    import shutil
    import subprocess
    import sys

    faulthandler.disable()
    builtins.exit = None
    builtins.quit = None

    os.environ["OMP_NUM_THREADS"] = "1"

    os.kill = None
    os.system = None
    os.putenv = None
    os.remove = None
    os.removedirs = None
    os.rmdir = None
    os.fchdir = None
    os.setuid = None
    os.fork = None
    os.forkpty = None
    os.killpg = None
    os.rename = None
    os.renames = None
    os.truncate = None
    os.replace = None
    os.unlink = None
    os.fchmod = None
    os.fchown = None
    os.chmod = None
    os.chown = None
    os.chroot = None
    os.fchdir = None
    os.lchflags = None
    os.lchmod = None
    os.lchown = None
    os.getcwd = None
    os.chdir = None
    builtins.open = None

    shutil.rmtree = None
    shutil.move = None
    shutil.chown = None

    subprocess.Popen = None  # type: ignore[assignment]

    __builtins__["help"] = None
    sys.modules["ipdb"] = None
    sys.modules["joblib"] = None
    sys.modules["resource"] = None
    sys.modules["psutil"] = None
    sys.modules["tkinter"] = None


def _surface_Area(base_edge, height):
    slant_height = math.sqrt((base_edge / 2) ** 2 + height**2)
    base_area = base_edge**2
    lateral_area = 4 * (base_edge * slant_height) / 2
    return round(base_area + lateral_area)


def _digit_distance_nums(num1, num2):
    str_num1, str_num2 = str(num1), str(num2)
    max_length = max(len(str_num1), len(str_num2))
    str_num1 = str_num1.zfill(max_length)
    str_num2 = str_num2.zfill(max_length)
    return sum(abs(int(digit1) - int(digit2)) for digit1, digit2 in zip(str_num1, str_num2))


def _poly(xs: list, x: float):
    return sum(coeff * math.pow(x, index) for index, coeff in enumerate(xs))


def _is_floats(value: Any) -> bool:
    if isinstance(value, float):
        return True
    if isinstance(value, (list, tuple)) and value:
        return all(isinstance(item, float) for item in value)
    if isinstance(value, np.ndarray):
        return value.dtype in (np.float32, np.float64)
    return False


def _prompt_to_source(prompt: Any) -> str:
    if isinstance(prompt, str):
        extracted = extract_code(prompt, preserve_indentation=True)
        return extracted if extracted else prompt.strip()

    if isinstance(prompt, list):
        for message in reversed(prompt):
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if isinstance(content, str):
                extracted = extract_code(content, preserve_indentation=True)
                if extracted:
                    return extracted
                if content.strip():
                    return content.strip()
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and isinstance(part.get("text"), str):
                        extracted = extract_code(part["text"], preserve_indentation=True)
                        if extracted:
                            return extracted
                        if part["text"].strip():
                            return part["text"].strip()

    raise ValueError("unable to extract EvalPlus prompt source from benchmark row")


def _mbpp_task_number(task_id: Any) -> int:
    return int(str(task_id).split("/")[-1])


def _mbpp_deserialize_inputs(task_id: Any, inputs: list) -> list:
    task_number = _mbpp_task_number(task_id)

    if task_number in [
        2,
        116,
        132,
        143,
        222,
        261,
        273,
        394,
        399,
        421,
        424,
        429,
        470,
        560,
        579,
        596,
        616,
        630,
        726,
        740,
        744,
        809,
    ]:
        return [[tuple(item) for item in inp] for inp in inputs]

    if task_number in [
        63,
        64,
        70,
        94,
        120,
        237,
        272,
        299,
        400,
        409,
        417,
        438,
        473,
        614,
        780,
    ]:
        return [[[tuple(item) for item in item_list] for item_list in inp] for inp in inputs]

    if task_number in [75, 413, 444, 753]:
        return [[[tuple(item) for item in inp[0]]] + [inp[1]] for inp in inputs]

    if task_number in [106, 750]:
        return [[inp[0]] + [tuple(inp[1])] for inp in inputs]

    if task_number == 115:
        return [
            [
                [
                    set(item) if isinstance(item, list) and len(item) else {}
                    for item in inp[0]
                ]
            ]
            for inp in inputs
        ]

    if task_number == 124:
        return [(float(inp[0]), complex(inp[1])) for inp in inputs]

    if task_number in [250, 405, 446, 617, 720, 763, 808]:
        return [[tuple(inp[0])] + [inp[1]] for inp in inputs]

    if task_number in [259, 401, 445]:
        converted = [[[tuple(item) for item in item_list] for item_list in inp] for inp in inputs]
        return [[tuple(item) for item in inp] for inp in converted]

    if task_number == 278:
        converted = [
            [[tuple(item) if isinstance(item, list) else item for item in inp[0]]]
            for inp in inputs
        ]
        return [[tuple(item) for item in inp] for inp in converted]

    if task_number == 307:
        return [[tuple(inp[0])] + [inp[1], inp[2]] for inp in inputs]

    if task_number == 722:
        return [[{key: tuple(value) for key, value in inp[0].items()}] + inp[1:] for inp in inputs]

    if task_number == 252:
        return [[complex(inp[0])] for inp in inputs]

    if task_number in [580, 615, 791]:

        def turn_all_list_into_tuple(inp):
            if isinstance(inp, list):
                return tuple(turn_all_list_into_tuple(item) for item in inp)
            return inp

        return [turn_all_list_into_tuple(inp) for inp in inputs]

    return inputs


def _infer_dataset(problem: dict[str, Any]) -> str:
    benchmark_name = str(problem.get("benchmark", "")).lower()
    task_id = str(problem.get("task_id", "")).lower()
    if "mbpp" in benchmark_name or task_id.startswith("mbpp/"):
        return "mbpp"
    if "human" in benchmark_name or task_id.startswith("humaneval/"):
        return "humaneval"
    raise ValueError(f"unsupported EvalPlus problem type: {problem.get('task_id')!r}")


def _normalize_problem(problem: dict[str, Any]) -> dict[str, Any]:
    dataset = _infer_dataset(problem)
    base_input = deepcopy(problem["base_input"])
    plus_input = deepcopy(problem["plus_input"])
    if dataset == "mbpp":
        base_input = _mbpp_deserialize_inputs(problem["task_id"], base_input)
        plus_input = _mbpp_deserialize_inputs(problem["task_id"], plus_input)

    return {
        "task_id": problem["task_id"],
        "dataset": dataset,
        "prompt": _prompt_to_source(problem["prompt"]),
        "canonical_solution": problem["canonical_solution"],
        "entry_point": problem["entry_point"],
        "base_input": base_input,
        "plus_input": plus_input,
        "atol": float(problem.get("atol", 0) or 0),
    }


def _has_entry_point_definition(code: str, entry_point: str) -> bool:
    try:
        tree = ast.parse(code)
    except Exception:
        return False

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == entry_point:
            return True
    return False


def _build_solution(problem: dict[str, Any], generation: str) -> str:
    code = generation or ""
    if (
        problem["dataset"] == "humaneval"
        and not _has_entry_point_definition(code, problem["entry_point"])
    ):
        return problem["prompt"].rstrip() + "\n" + code.strip("\n")
    return code


def _trusted_exec(
    code: str,
    inputs: list,
    entry_point: str,
    record_time: bool = False,
    output_not_none: bool = False,
):
    exec_globals: dict[str, Any] = {}
    exec(code, exec_globals)
    fn = exec_globals[entry_point]

    results = []
    runtimes = []
    for inp in inputs:
        inp = deepcopy(inp)
        if record_time:
            start = time.time()
            results.append(fn(*inp))
            runtimes.append(time.time() - start)
        else:
            results.append(fn(*inp))

    if output_not_none:
        results = [item is not None for item in results]

    if record_time:
        return results, runtimes
    return results


def _build_expected_output(problem: dict[str, Any]) -> dict[str, Any]:
    code = problem["prompt"] + problem["canonical_solution"]
    output_not_none = (
        problem["dataset"] == "mbpp"
        and problem["entry_point"] in MBPP_OUTPUT_NOT_NONE_TASKS
    )
    base_output, base_time = _trusted_exec(
        code,
        problem["base_input"],
        problem["entry_point"],
        record_time=True,
        output_not_none=output_not_none,
    )
    plus_output, plus_time = _trusted_exec(
        code,
        problem["plus_input"],
        problem["entry_point"],
        record_time=True,
        output_not_none=output_not_none,
    )
    return {
        "base": base_output,
        "base_time": base_time,
        "plus": plus_output,
        "plus_time": plus_time,
    }


def _record_error(error_queue: multiprocessing.Queue, exc: BaseException):
    try:
        error_queue.put_nowait(repr(exc))
    except Exception:
        pass


def _unsafe_execute(
    dataset: str,
    entry_point: str,
    code: str,
    inputs: list,
    expected: list,
    time_limits: list[float],
    atol: float,
    fast_check: bool,
    stat: Value,
    details: Array,
    progress: Value,
    error_queue: multiprocessing.Queue,
):
    with create_tempdir():
        import shutil

        rmtree = shutil.rmtree
        rmdir = os.rmdir
        chdir_impl = os.chdir
        reliability_guard(maximum_memory_bytes=_query_maximum_memory_bytes())
        exec_globals: dict[str, Any] = {}
        try:
            with swallow_io():
                exec(code, exec_globals)
                fn = exec_globals[entry_point]

            for index, inp in enumerate(inputs):
                try:
                    with time_limit(time_limits[index]):
                        with swallow_io():
                            output = fn(*inp)

                    expected_output = expected[index]
                    exact_match = output == expected_output

                    if dataset == "mbpp":
                        if entry_point == "are_equivalent":
                            exact_match = exact_match or True
                        elif entry_point == "sum_div":
                            exact_match = exact_match or output == 0
                        elif entry_point == "surface_Area":
                            exact_match = (
                                exact_match
                                or abs(output - _surface_Area(*inp)) <= atol
                            )
                        elif entry_point == "digit_distance_nums":
                            exact_match = exact_match or output == _digit_distance_nums(*inp)
                        elif entry_point in MBPP_OUTPUT_SET_EQ_TASKS:
                            exact_match = set(output) == set(expected_output)
                        elif entry_point in MBPP_OUTPUT_NOT_NONE_TASKS:
                            if isinstance(output, bool):
                                exact_match = output == expected_output
                            else:
                                exact_match = expected_output == (output is not None)

                    if dataset == "humaneval" and entry_point == "find_zero":
                        assert abs(_poly(*inp, output)) <= atol
                        details[index] = True
                        progress.value += 1
                        continue

                    atol_value = atol
                    if atol_value == 0 and _is_floats(expected_output):
                        atol_value = 1e-6

                    if not exact_match and atol_value != 0:
                        assert type(output) == type(expected_output)
                        if isinstance(expected_output, (list, tuple)):
                            assert len(output) == len(expected_output)
                        assert np.allclose(output, expected_output, rtol=1e-07, atol=atol_value)
                    else:
                        assert exact_match
                except BaseException as exc:
                    details[index] = False
                    progress.value += 1
                    _record_error(error_queue, exc)
                    if fast_check:
                        raise
                    continue

                details[index] = True
                progress.value += 1

            stat.value = _SUCCESS
        except TimeoutException as exc:
            _record_error(error_queue, exc)
            stat.value = _TIMEOUT
        except BaseException as exc:
            _record_error(error_queue, exc)
            stat.value = _FAILED
        finally:
            shutil.rmtree = rmtree
            os.rmdir = rmdir
            os.chdir = chdir_impl


def _untrusted_check(
    dataset: str,
    code: str,
    inputs: list,
    entry_point: str,
    expected: list,
    atol: float,
    ref_time: list[float],
    timeout_cap: Optional[float],
    fast_check: bool = True,
    min_time_limit: float = DEFAULT_MIN_TIME_LIMIT,
    gt_time_limit_factor: float = DEFAULT_GT_TIME_LIMIT_FACTOR,
):
    time_limits = [max(min_time_limit, gt_time_limit_factor * t) for t in ref_time]
    max_timeout = 60 if timeout_cap is None else float(timeout_cap)
    timeout = min(max_timeout, sum(time_limits)) + 1
    if not fast_check:
        timeout += 1

    progress = Value("i", 0)
    stat = Value("i", _UNKNOWN)
    details = Array("b", [False for _ in range(len(inputs))])
    error_queue: multiprocessing.Queue = multiprocessing.Queue()

    process = multiprocessing.Process(
        target=_unsafe_execute,
        args=(
            dataset,
            entry_point,
            code,
            inputs,
            expected,
            time_limits,
            atol,
            fast_check,
            stat,
            details,
            progress,
            error_queue,
        ),
    )
    process.start()
    process.join(timeout=timeout + 1)
    if process.is_alive():
        process.terminate()
        time.sleep(0.1)
    if process.is_alive():
        process.kill()
        time.sleep(0.1)

    status = _STATUS_MAPPING[stat.value]
    bool_details = [bool(item) for item in details[: progress.value]]

    if not status:
        status = TIMEOUT
    if status == PASS and (len(bool_details) != len(inputs) or not all(bool_details)):
        status = FAIL

    error = None
    while True:
        try:
            candidate = error_queue.get_nowait()
        except queue.Empty:
            break
        if candidate and error is None:
            error = candidate

    process.close()
    error_queue.close()
    error_queue.join_thread()
    return status, bool_details, error


def _summarize_metadata(
    base_status: str,
    plus_status: str,
    base_error: Optional[str],
    plus_error: Optional[str],
) -> dict[str, Any]:
    overall_status = PASS if base_status == plus_status == PASS else plus_status
    if base_status != PASS:
        overall_status = base_status

    metadata: dict[str, Any] = {"status": overall_status}
    if base_status != PASS or plus_status != PASS:
        metadata["base_status"] = base_status
        metadata["plus_status"] = plus_status
        error = base_error if base_status != PASS else plus_error
        if error:
            metadata["error"] = error
    return metadata


def _evaluate_generation(
    problem: dict[str, Any],
    generation: str,
    expected_output: dict[str, Any],
    timeout_cap: Optional[float],
):
    code = _build_solution(problem, generation)
    base_status, _, base_error = _untrusted_check(
        problem["dataset"],
        code,
        problem["base_input"],
        problem["entry_point"],
        expected=expected_output["base"],
        atol=problem["atol"],
        ref_time=expected_output["base_time"],
        timeout_cap=timeout_cap,
    )
    plus_status, _, plus_error = _untrusted_check(
        problem["dataset"],
        code,
        problem["plus_input"],
        problem["entry_point"],
        expected=expected_output["plus"],
        atol=problem["atol"],
        ref_time=expected_output["plus_time"],
        timeout_cap=timeout_cap,
    )

    return [
        base_status == PASS,
        plus_status == PASS,
    ], _summarize_metadata(base_status, plus_status, base_error, plus_error)


def _evaluate_generations_by_problem(args):
    problem_generations, problem, expected_output, timeout_cap = args

    results = []
    metadata = []
    for generation in problem_generations:
        current_results, current_metadata = _evaluate_generation(
            problem=problem,
            generation=generation,
            expected_output=expected_output,
            timeout_cap=timeout_cap,
        )
        results.append(current_results)
        metadata.append(current_metadata)
    return results, metadata


def _compute_pass_at_k_metrics(
    correctness_by_problem: list[list[bool]],
    k_list: list[int],
) -> dict[str, Any]:
    total = np.array([len(item) for item in correctness_by_problem], dtype=int)
    correct = np.array([sum(item) for item in correctness_by_problem], dtype=int)

    detail = {
        f"pass@{k}": estimate_pass_at_k(total, correct, k).tolist()
        for k in k_list
        if len(total) and (total >= k).all()
    }
    metrics = {
        f"pass@{k}": float(estimate_pass_at_k(total, correct, k).mean())
        for k in k_list
        if len(total) and (total >= k).all()
    }
    metrics["detail"] = {
        key: dict(enumerate(values))
        for key, values in detail.items()
    }
    return metrics


def evalplus_metrics(
    problems_list: list[dict],
    generations_list: list[list[str]],
    k_list=None,
    num_process_evaluate: int = 16,
    timeout: int = 10,
):
    if len(problems_list) != len(generations_list):
        raise ValueError("problems_list and generations_list must have the same length")

    if k_list is None:
        k_list = [1, 10, 100]

    normalized_problems = [_normalize_problem(problem) for problem in problems_list]
    expected_outputs = [_build_expected_output(problem) for problem in normalized_problems]

    inputs = [
        (
            generations_list[index],
            normalized_problems[index],
            expected_outputs[index],
            timeout,
        )
        for index in range(len(normalized_problems))
    ]

    results: dict[int, list[list[bool]]] = {}
    metadatas: dict[int, list[dict[str, Any]]] = {}

    if num_process_evaluate <= 1:
        with tqdm(total=len(inputs)) as progress:
            for index, args in enumerate(inputs):
                results[index], metadatas[index] = _evaluate_generations_by_problem(args)
                progress.update(1)
    else:
        with tqdm(total=len(inputs)) as progress:
            with ProcessPoolExecutor(max_workers=num_process_evaluate) as executor:
                futures = {
                    executor.submit(_evaluate_generations_by_problem, args): index
                    for index, args in enumerate(inputs)
                }
                for future in as_completed(futures):
                    index = futures[future]
                    results[index], metadatas[index] = future.result()
                    progress.update(1)

    ordered_results = [results[index] for index in range(len(normalized_problems))]
    ordered_metadatas = [metadatas[index] for index in range(len(normalized_problems))]

    base_correctness = [
        [sample_result[0] for sample_result in problem_results]
        for problem_results in ordered_results
    ]
    plus_correctness = [
        [all(sample_result) for sample_result in problem_results]
        for problem_results in ordered_results
    ]

    base_metrics = _compute_pass_at_k_metrics(base_correctness, list(k_list))
    plus_metrics = _compute_pass_at_k_metrics(plus_correctness, list(k_list))

    metrics = dict(plus_metrics)
    metrics["base"] = base_metrics
    metrics["plus"] = plus_metrics

    return [metrics, defaultdict(list, results), ordered_metadatas]
