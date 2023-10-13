import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

"""
Model and Tokenizer V1: trained with 1000 rows 
-> question_generator/model/v1 
-> question_generator/tokenizer/v1

Model and Tokenizer V2: trained with 3000 rows
-> question_generator/model/v2
-> question_generator/tokenizer/v2

Model and Tokenizer V3: trained with all rows from datasets (78664 rows-train, 9652 rows-validation)
-> question_generator/model/v3
-> question_generator/tokenizer/v3
"""


def question_parser(question: str) -> str:
    return " ".join(question.split(":")[1].split())


def generate_questions(
    context: str, answer: str, n_questions: int = 1, model_version: str = "v1"
):
    trained_model_path = f"question_generator/model/{model_version}"
    trained_tokenizer = f"question_generator/tokenizer/{model_version}"

    model = T5ForConditionalGeneration.from_pretrained(trained_model_path)
    tokenizer = T5Tokenizer.from_pretrained(trained_tokenizer)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print("device ", device)
    model = model.to(device)

    text = "context: " + context + " " + "answer: " + answer + " </s>"
    print(text)

    encoding = tokenizer.encode_plus(
        text, max_length=512, padding=True, return_tensors="pt"
    )
    input_ids, attention_mask = encoding["input_ids"].to(device), encoding[
        "attention_mask"
    ].to(device)

    model.eval()
    beam_outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=72,
        early_stopping=True,
        num_beams=5,
        num_return_sequences=n_questions,
    )

    questions = []

    for beam_output in beam_outputs:
        sent = tokenizer.decode(
            beam_output, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        print(sent)
        questions.append(question_parser(sent))

    return questions


"""
context = "President Donald Trump said and predicted that some states would reopen this month."
answer = "Donald Trump"

questions = generate_questions(context, answer, 1, "v2")
print(questions)
"""
