from __future__ import annotations


def extract_code(model_output: str | None, preserve_indentation: bool = False) -> str:
    if not model_output:
        return ""

    output_lines = model_output.splitlines()
    fence_lines = [idx for idx, line in enumerate(output_lines) if "```" in line]
    if len(fence_lines) < 2:
        return ""

    start = fence_lines[-2] + 1
    end = fence_lines[-1]
    code = "\n".join(output_lines[start:end])
    if preserve_indentation:
        return code.strip("\n")
    return code.strip()
