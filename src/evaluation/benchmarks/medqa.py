"""
MedQA Benchmark

Medical Q&A problems for evaluating medical knowledge reasoning ability.
"""

from typing import List, Tuple


def load_medqa(subset: str = "us") -> Tuple[List[str], List[str]]:
    """
    Load MedQA dataset.

    Args:
        subset: subset ("us", "mainland", "taiwan")

    Returns:
        (questions, answers) tuple
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

            # Build complete question
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
    """Return sample MedQA data"""
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
