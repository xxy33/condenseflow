"""
MBPP (Mostly Basic Python Problems) Benchmark

Python编程题目，用于评估代码生成能力。
"""

from typing import List, Tuple


def load_mbpp(split: str = "test") -> Tuple[List[str], List[str]]:
    """
    加载MBPP数据集。

    Args:
        split: 数据集分割 ("test", "train", "validation")

    Returns:
        (questions, answers) 元组
    """
    try:
        from datasets import load_dataset
        dataset = load_dataset("mbpp", split=split)

        questions = []
        answers = []

        for item in dataset:
            prompt = item["text"]
            code = item["code"]

            # 添加测试用例信息
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
    """返回示例MBPP数据"""
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
    评估生成的代码是否正确。

    Args:
        prediction: 生成的代码
        test_cases: 测试用例列表

    Returns:
        是否通过所有测试
    """
    try:
        # 创建执行环境
        exec_globals = {}
        exec(prediction, exec_globals)

        # 运行测试用例
        for test in test_cases:
            exec(test, exec_globals)

        return True
    except Exception:
        return False
