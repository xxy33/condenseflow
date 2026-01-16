"""
MBPP (Mostly Basic Python Problems) Benchmark

Python programming problems for evaluating code generation ability.
"""

from typing import List, Tuple


def load_mbpp(split: str = "test") -> Tuple[List[str], List[str]]:
    """
    Load MBPP dataset.

    Args:
        split: dataset split ("test", "train", "validation")

    Returns:
        (questions, answers) tuple
    """
    try:
        from datasets import load_dataset
        dataset = load_dataset("mbpp", split=split)

        questions = []
        answers = []

        for item in dataset:
            prompt = item["text"]
            code = item["code"]

            # Add test case information
            test_cases = item.get("test_list", [])
            if test_cases:
                prompt += "\n\nTest cases:\n" + "\n".join(test_cases[:3])

            questions.append(prompt)
            answers.append(code)

        return questions, answers

    except Exception as e:
        print(f"Failed to load MBPP from HuggingFace: {e}")
        return _get_sample_mbpp()


def _get_sample_mbpp() -> Tuple[List[str], List[str]]:
    """Return sample MBPP data"""
    questions = [
        """Write a function to find the maximum of two numbers.

Test cases:
assert max_of_two(5, 10) == 10
assert max_of_two(-1, -5) == -1
assert max_of_two(0, 0) == 0""",
    ]
    answers = [
        """def max_of_two(a, b):
    if a > b:
        return a
    return b""",
    ]
    return questions, answers


def evaluate_code(prediction: str, test_cases: List[str]) -> bool:
    """
    Evaluate whether generated code is correct.

    Args:
        prediction: generated code
        test_cases: list of test cases

    Returns:
        whether all tests pass
    """
    try:
        # Create execution environment
        exec_globals = {}
        exec(prediction, exec_globals)

        # Run test cases
        for test in test_cases:
            exec(test, exec_globals)

        return True
    except Exception:
        return False
