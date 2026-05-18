from __future__ import annotations

from typing import Any, Dict, List

LCB_SYSTEM_MESSAGE_GENERIC = (
    "You are an expert Python programmer. You will be given a question "
    "(problem specification) and will generate a correct Python program "
    "that matches the specification and passes all tests."
)

LCB_FORMATTING_MESSAGE_WITH_STARTER_CODE = (
    "You will use the following starter code to write the solution to the "
    "problem and enclose your code within delimiters."
)

LCB_FORMATTING_WITHOUT_STARTER_CODE = (
    "Read the inputs from stdin solve the problem and write the answer to "
    "stdout (do not directly test on the sample inputs). Enclose your code "
    "within delimiters as follows. Ensure that when the python program runs, "
    "it reads the inputs, runs the algorithm and writes output to STDOUT."
)

EVAL_PLUS_INSTRUCTION_PREFIX = "Please provide a self-contained Python script that solves the following problem in a markdown code block:"

def build_livecodebench_user_prompt(
    question_content: str,
    starter_code: str,
) -> str:
    prompt = f"### Question:\n{question_content}\n\n"
    if starter_code:
        prompt += f"### Format: {LCB_FORMATTING_MESSAGE_WITH_STARTER_CODE}\n"
        prompt += f"```python\n{starter_code}\n```\n\n"
    else:
        prompt += f"### Format: {LCB_FORMATTING_WITHOUT_STARTER_CODE}\n"
        prompt += "```python\n# YOUR CODE HERE\n```\n\n"
    prompt += "### Answer: (use the provided format with backticks)\n\n"
    return prompt


def build_livecodebench_messages(
    question_content: str,
    starter_code: str,
) -> List[Dict[str, Any]]:
    return [
        {"role": "system", "content": LCB_SYSTEM_MESSAGE_GENERIC},
            {
                "role": "user",
            "content": build_livecodebench_user_prompt(
                question_content=question_content,
                starter_code=starter_code,
            ),
        },
    ]

def build_evalplus_messages(task_prompt: str) -> List[Dict[str, Any]]:
    prompt = EVAL_PLUS_INSTRUCTION_PREFIX + f"\n```python\n{task_prompt.strip()}\n```"
    return [{"role": "user", "content": prompt}]
