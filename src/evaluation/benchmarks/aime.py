"""
AIME (American Invitational Mathematics Examination) Benchmark

数学竞赛题目，用于评估数学推理能力。
"""

from typing import List, Tuple, Optional


def load_aime(year: Optional[int] = None) -> Tuple[List[str], List[str]]:
    """
    加载AIME数据集。

    Args:
        year: 指定年份，None则加载所有

    Returns:
        (questions, answers) 元组
    """
    # 示例数据 - 实际使用时应从数据文件加载
    aime_data = {
        2024: [
            {
                "question": "Find the number of ways to place 4 rooks on a 4x4 chessboard such that no two rooks attack each other.",
                "answer": "24"
            },
            {
                "question": "Let $a$ and $b$ be positive integers such that $a^2 + b^2 = 2024$. Find the sum of all possible values of $a + b$.",
                "answer": "88"
            },
        ],
        2025: [
            {
                "question": "Find the remainder when $2^{2025} + 3^{2025}$ is divided by 7.",
                "answer": "0"
            },
        ],
    }

    questions = []
    answers = []

    if year is not None:
        data = aime_data.get(year, [])
        for item in data:
            questions.append(item["question"])
            answers.append(item["answer"])
    else:
        for year_data in aime_data.values():
            for item in year_data:
                questions.append(item["question"])
                answers.append(item["answer"])

    return questions, answers


def load_aime_from_file(file_path: str) -> Tuple[List[str], List[str]]:
    """从文件加载AIME数据"""
    import json

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    questions = [item["question"] for item in data]
    answers = [item["answer"] for item in data]

    return questions, answers
