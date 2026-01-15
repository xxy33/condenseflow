"""
MedQA Benchmark

医学问答题目，用于评估医学知识推理能力。
"""

from typing import List, Tuple


def load_medqa(subset: str = "us") -> Tuple[List[str], List[str]]:
    """
    加载MedQA数据集。

    Args:
        subset: 子集 ("us", "mainland", "taiwan")

    Returns:
        (questions, answers) 元组
    """
    try:
        from datasets import load_dataset

        if subset == "us":
            dataset = load_dataset("bigbio/med_qa", "med_qa_en_source", split="test")
        else:
            dataset = load_dataset("bigbio/med_qa", f"med_qa_{subset}_source", split="test")

        questions = []
        answers = []

        for item in dataset:
            question = item["question"]
            options = item.get("options", {})

            # 构建完整问题
            full_question = question + "\n\n"
            for key, value in options.items():
                full_question += f"{key}) {value}\n"

            questions.append(full_question)
            answers.append(item["answer_idx"])

        return questions, answers

    except Exception as e:
        print(f"Failed to load MedQA from HuggingFace: {e}")
        return _get_sample_medqa()


def _get_sample_medqa() -> Tuple[List[str], List[str]]:
    """返回示例MedQA数据"""
    questions = [
        """A 45-year-old man presents with chest pain that worsens with deep breathing.
Physical examination reveals a friction rub. Which of the following is the most likely diagnosis?

A) Myocardial infarction
B) Pericarditis
C) Pulmonary embolism
D) Pneumothorax""",
    ]
    answers = ["B"]
    return questions, answers
