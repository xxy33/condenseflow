"""
GPQA (Graduate-Level Google-Proof Q&A) Benchmark

研究生级别的科学问答题目。
"""

from typing import List, Tuple


def load_gpqa(subset: str = "diamond") -> Tuple[List[str], List[str]]:
    """
    加载GPQA数据集。

    Args:
        subset: 子集名称 ("diamond", "extended", "main")

    Returns:
        (questions, answers) 元组
    """
    try:
        from datasets import load_dataset
        dataset = load_dataset("Idavidrein/gpqa", subset, split="train")

        questions = []
        answers = []

        for item in dataset:
            question = item["Question"]
            # 添加选项
            choices = [
                f"A) {item['Correct Answer']}",
                f"B) {item['Incorrect Answer 1']}",
                f"C) {item['Incorrect Answer 2']}",
                f"D) {item['Incorrect Answer 3']}",
            ]
            full_question = f"{question}\n\n" + "\n".join(choices)
            questions.append(full_question)
            answers.append("A")  # 正确答案总是A（需要打乱）

        return questions, answers

    except Exception as e:
        print(f"Failed to load GPQA from HuggingFace: {e}")
        # 返回示例数据
        return _get_sample_gpqa()


def _get_sample_gpqa() -> Tuple[List[str], List[str]]:
    """返回示例GPQA数据"""
    questions = [
        """Which of the following best describes the mechanism of action of CRISPR-Cas9?

A) It uses guide RNA to direct the Cas9 nuclease to specific DNA sequences for cleavage
B) It uses random mutagenesis to introduce changes in the genome
C) It relies on homologous recombination without any nuclease activity
D) It uses zinc finger proteins to bind and modify DNA""",
    ]
    answers = ["A"]
    return questions, answers
