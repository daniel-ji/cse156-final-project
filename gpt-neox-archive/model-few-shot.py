from transformers import GPTNeoXForQuestionAnswering, GPTNeoXTokenizerFast
from datasets import load_dataset
import torch
import numpy as np

model_name = "EleutherAI/gpt-neox-20b"
model = GPTNeoXForQuestionAnswering.from_pretrained(model_name, torch_dtype=torch.float16)
tokenizer = GPTNeoXTokenizerFast.from_pretrained(model_name)

dataset = load_dataset("aps/super_glue", "record")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

def find_best_answer_span(start_logits, end_logits, max_answer_length=8):
    best_start, best_end = 0, 0
    max_score = -np.inf

    # Iterate over all start positions
    for start_index in range(len(start_logits)):
        for end_index in range(start_index, min(start_index + max_answer_length, len(end_logits))):
            # Ensure the end index is not less than the start index
            if end_index < start_index:
                continue
            # Calculate score as the sum of the start and end logits
            score = start_logits[start_index] + end_logits[end_index]
            if score > max_score:
                max_score = score
                best_start, best_end = start_index, end_index

    return best_start, best_end


def generate_answer(query, context):
    inputs = tokenizer(query, context, return_tensors="pt").to(device)
    outputs = model(**inputs)
    answer_start_index, answer_end_index = find_best_answer_span(outputs.start_logits[0], outputs.end_logits[0])
    predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
    return tokenizer.decode(predict_answer_tokens, skip_special_tokens=True)

train_counter = 0
FEW_SHOT_COUNT = 3

for example in dataset["test"]:
    few_shot_context = ""
    for i in range(FEW_SHOT_COUNT):
        few_shot_context += "CONTEXT: \n\n" + dataset["train"][train_counter]["passage"] + " \n\n QUERY: \n\n" + dataset["train"][train_counter]["query"] + " What is the value of @placeholder, based on teh given text? \n\n ANSWER: \n\n" + dataset["train"][train_counter]["answers"]["text"][0] + "\n\n\n\n"
        train_counter += 1

    query = "\n\n QUERY: \n\n" + example["query"] + " What is the value of @placeholder, based on the given text?"
    context = "CONTEXT: \n\n" + example["passage"]
    answer = generate_answer(query, context)
    print(context)
    print("\n\n\n\n\n")
    print("Query: ", query)
    print(answer)
    print("\n\n\n\n\n")
    print("--------------------------------------------------------------------------------------")