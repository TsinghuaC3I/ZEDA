from __future__ import annotations

import ast
import faulthandler
import json
import platform
import signal
import sys
import time
from datetime import datetime
from decimal import Decimal
from enum import Enum
from io import StringIO
from types import ModuleType
from unittest.mock import mock_open, patch


import numpy as np


IMPORT_STRING = (
    "from string import *\nfrom re import *\nfrom datetime import *\n"
    "from collections import *\nfrom heapq import *\nfrom bisect import *\n"
    "from copy import *\nfrom math import *\nfrom random import *\n"
    "from statistics import *\nfrom itertools import *\n"
    "from functools import *\nfrom operator import *\nfrom io import *\n"
    "from sys import *\nfrom json import *\nfrom builtins import *\n"
    "from typing import *\nimport string\nimport re\nimport datetime\n"
    "import collections\nimport heapq\nimport bisect\nimport copy\n"
    "import math\nimport random\nimport statistics\nimport itertools\n"
    "import functools\nimport operator\nimport io\nimport sys\nimport json\n"
    "sys.setrecursionlimit(50000)\n"
)


def truncatefn(value, length=300):
    if not isinstance(value, str):
        value = str(value)
    if len(value) <= length:
        return value
    return value[: length // 2] + "...(truncated) ..." + value[-length // 2 :]


class CodeType(Enum):
    CALL_BASED = 0
    STANDARD_INPUT = 1


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException


class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        self._stringio = StringIO()
        self._stringio.close = lambda x: 1
        sys.stdout = self._stringio
        return self

    def __exit__(self, *args):
        self.append(self._stringio.getvalue())
        del self._stringio
        sys.stdout = self._stdout


class MockBuffer:
    def __init__(self, inputs: str):
        self.inputs = inputs.encode("utf-8")

    def read(self, *args):
        return self.inputs

    def readline(self, *args):
        return self.inputs.split(b"\n")[0] + b"\n"


class MockStdinWithBuffer:
    def __init__(self, inputs: str):
        self.inputs = inputs
        self._stringio = StringIO(inputs)
        self.buffer = MockBuffer(inputs)

    def read(self, *args):
        return self.inputs

    def readline(self, *args):
        return self._stringio.readline(*args)

    def readlines(self, *args):
        return self.inputs.split("\n")

    def __getattr__(self, name):
        return getattr(self._stringio, name)


def clean_if_name(code: str) -> str:
    try:
        tree = ast.parse(code)
        last_block = tree.body[-1]
        if isinstance(last_block, ast.If):
            condition = last_block.test
            if ast.unparse(condition).strip() == "__name__ == '__main__'":
                return (
                    ast.unparse(tree.body[:-1]) + "\n" + ast.unparse(last_block.body)
                )
    except Exception:
        return code
    return code


def make_function(code: str) -> str:
    try:
        import_stmts = []
        other_stmts = []
        tree = ast.parse(code)
        for stmt in tree.body:
            if isinstance(stmt, (ast.Import, ast.ImportFrom)):
                import_stmts.append(stmt)
            else:
                other_stmts.append(stmt)

        function_ast = ast.FunctionDef(
            name="wrapped_function",
            args=ast.arguments(
                posonlyargs=[],
                args=[],
                kwonlyargs=[],
                kw_defaults=[],
                defaults=[],
            ),
            body=other_stmts,
            decorator_list=[],
            lineno=-1,
        )
        return (
            IMPORT_STRING
            + "\n"
            + ast.unparse(import_stmts)
            + "\n"
            + ast.unparse(function_ast)
        )
    except Exception:
        return code


def call_method(method, inputs):
    if isinstance(inputs, list):
        inputs = "\n".join(inputs)

    inputs_line_iterator = iter(inputs.split("\n"))
    mock_stdin = MockStdinWithBuffer(inputs)

    @patch("builtins.open", mock_open(read_data=inputs))
    @patch("sys.stdin", mock_stdin)
    @patch("sys.stdin.readline", lambda *args: next(inputs_line_iterator))
    @patch("sys.stdin.readlines", lambda *args: inputs.split("\n"))
    @patch("sys.stdin.read", lambda *args: inputs)
    def _inner_call_method(_method):
        try:
            return _method()
        except SystemExit:
            return None

    return _inner_call_method(method)


def get_function(compiled_sol, fn_name: str):
    try:
        assert hasattr(compiled_sol, fn_name)
        return getattr(compiled_sol, fn_name)
    except Exception:
        return None


def compile_code(code: str, timeout: int):
    signal.alarm(timeout)
    try:
        tmp_sol = ModuleType("tmp_sol", "")
        exec(code, tmp_sol.__dict__)
        if "class Solution" in code:
            compiled_sol = tmp_sol.Solution()
        else:
            compiled_sol = tmp_sol
        assert compiled_sol is not None
        return compiled_sol
    finally:
        signal.alarm(0)


def convert_line_to_decimals(line: str) -> tuple[bool, list[Decimal]]:
    try:
        return True, [Decimal(elem) for elem in line.split()]
    except Exception:
        return False, []


def get_stripped_lines(value: str):
    return [line.strip() for line in value.strip().split("\n")]


def grade_call_based(
    code: str,
    all_inputs: list,
    all_outputs: list,
    fn_name: str,
    timeout: int,
):
    code = IMPORT_STRING + "\n\n" + code
    compiled_sol = compile_code(code, timeout)
    if compiled_sol is None:
        return None

    method = get_function(compiled_sol, fn_name)
    if method is None:
        return None

    all_inputs = [
        [json.loads(line) for line in inputs.split("\n")] for inputs in all_inputs
    ]
    all_outputs = [json.loads(output) for output in all_outputs]

    total_execution = 0
    all_results = []
    for gt_inp, gt_out in zip(all_inputs, all_outputs):
        signal.alarm(timeout)
        faulthandler.enable()
        try:
            start = time.time()
            prediction = method(*gt_inp)
            total_execution += time.time() - start
            signal.alarm(0)

            if isinstance(prediction, tuple):
                prediction = list(prediction)

            tmp_result = prediction == gt_out
            all_results.append(tmp_result)
            if not tmp_result:
                return all_results, {
                    "output": truncatefn(prediction),
                    "inputs": truncatefn(gt_inp),
                    "expected": truncatefn(gt_out),
                    "error_code": -2,
                    "error_message": "Wrong Answer",
                }
        except Exception as exc:
            signal.alarm(0)
            if "timeoutexception" in repr(exc).lower():
                all_results.append(-3)
                return all_results, {
                    "error": repr(exc),
                    "error_code": -3,
                    "error_message": "Time Limit Exceeded",
                    "inputs": truncatefn(gt_inp),
                    "expected": truncatefn(gt_out),
                }
            all_results.append(-4)
            return all_results, {
                "error": repr(exc),
                "error_code": -4,
                "error_message": "Runtime Error",
                "inputs": truncatefn(gt_inp),
                "expected": truncatefn(gt_out),
            }
        finally:
            signal.alarm(0)
            faulthandler.disable()

    return all_results, {"execution time": total_execution}


def grade_stdio(code: str, all_inputs: list, all_outputs: list, timeout: int):
    code = make_function(clean_if_name(code))
    compiled_sol = compile_code(code, timeout)
    if compiled_sol is None:
        return None

    method = get_function(compiled_sol, "wrapped_function")
    if method is None:
        return None

    all_results = []
    total_execution_time = 0
    for gt_inp, gt_out in zip(all_inputs, all_outputs):
        signal.alarm(timeout)
        faulthandler.enable()
        with Capturing() as captured_output:
            try:
                start = time.time()
                call_method(method, gt_inp)
                total_execution_time += time.time() - start
                signal.alarm(0)
            except Exception as exc:
                signal.alarm(0)
                if "timeoutexception" in repr(exc).lower():
                    all_results.append(-3)
                    return all_results, {
                        "error": repr(exc),
                        "error_code": -3,
                        "error_message": "Time Limit Exceeded",
                        "inputs": truncatefn(gt_inp),
                        "expected": truncatefn(gt_out),
                    }
                all_results.append(-4)
                return all_results, {
                    "error": repr(exc),
                    "error_code": -4,
                    "error_message": "Runtime Error",
                    "inputs": truncatefn(gt_inp),
                    "expected": truncatefn(gt_out),
                }
            finally:
                signal.alarm(0)
                faulthandler.disable()

        prediction = captured_output[0]
        predicted_lines = get_stripped_lines(prediction)
        expected_lines = get_stripped_lines(gt_out)
        wrong_answer = {
            "output": truncatefn(prediction),
            "inputs": truncatefn(gt_inp),
            "expected": truncatefn(gt_out),
            "error_code": -2,
        }

        if len(predicted_lines) != len(expected_lines):
            all_results.append(-2)
            wrong_answer["error_message"] = "Wrong answer: mismatched output length"
            return all_results, wrong_answer

        for idx, (predicted, expected) in enumerate(zip(predicted_lines, expected_lines)):
            wrong_answer["error_message"] = (
                f"Wrong answer at output_line_idx={idx}: "
                f"{truncatefn(predicted)} != {truncatefn(expected)}"
            )
            if predicted == expected:
                continue

            success, predicted_decimals = convert_line_to_decimals(predicted)
            if not success:
                all_results.append(-2)
                return all_results, wrong_answer

            success, expected_decimals = convert_line_to_decimals(expected)
            if not success:
                all_results.append(-2)
                return all_results, wrong_answer

            if predicted_decimals == expected_decimals:
                continue

            all_results.append(-2)
            return all_results, wrong_answer

        all_results.append(True)

    return all_results, {"execution time": total_execution_time}


def run_test(sample, test=None, debug=False, timeout=6):
    signal.signal(signal.SIGALRM, timeout_handler)
    reliability_guard()

    if debug:
        print(f"start = {datetime.now().time()}")

    in_outs = json.loads(sample["input_output"])
    if in_outs.get("fn_name") is None:
        which_type = CodeType.STANDARD_INPUT
        method_name = None
    else:
        which_type = CodeType.CALL_BASED
        method_name = in_outs["fn_name"]

    if test is None:
        raise AssertionError("should not happen: test code is none")

    if which_type == CodeType.CALL_BASED:
        signal.alarm(timeout)
        try:
            results, metadata = grade_call_based(
                code=test,
                all_inputs=in_outs["inputs"],
                all_outputs=in_outs["outputs"],
                fn_name=method_name,
                timeout=timeout,
            )
            return results, metadata
        except Exception as exc:
            return [-4], {
                "error_code": -4,
                "error_message": f"Error during testing: {exc}",
            }
        finally:
            signal.alarm(0)

    signal.alarm(timeout)
    try:
        results, metadata = grade_stdio(
            code=test,
            all_inputs=in_outs["inputs"],
            all_outputs=in_outs["outputs"],
            timeout=timeout,
        )
        return results, metadata
    except Exception as exc:
        return [-4], {
            "error_code": -4,
            "error_message": f"Error during testing: {exc}",
        }
    finally:
        signal.alarm(0)


def reliability_guard(maximum_memory_bytes=None):
    if maximum_memory_bytes is not None:
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

    faulthandler.disable()

    import builtins
    import os
    import shutil
    import subprocess

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

    shutil.rmtree = None
    shutil.move = None
    shutil.chown = None

    subprocess.Popen = None  # type: ignore

    __builtins__["help"] = None

    sys.modules["ipdb"] = None
    sys.modules["joblib"] = None
    sys.modules["resource"] = None
    sys.modules["psutil"] = None
    sys.modules["tkinter"] = None
