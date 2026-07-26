from nebius_poc.data import build_question

SUBJECT = "professional_law"


def record(question, choices, answer, subject=SUBJECT):
    return {"subject": subject, "question": question, "choices": list(choices), "answer": answer}


def question_from(question, choices, answer, subject=SUBJECT):
    return build_question(record(question, choices, answer, subject))
