from transformers import AutoModelForCausalLM, GPTNeoXTokenizerFast
from datasets import load_dataset
import torch

model_name = "EleutherAI/gpt-neox-20b"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
tokenizer = GPTNeoXTokenizerFast.from_pretrained(model_name)

dataset = load_dataset("aps/super_glue", "record")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

def find_best_answer_span(start_logits, end_logits, max_answer_length=8):
    best_start, best_end = 0, 0
    max_score = -np.inf

    # Convert logits to probabilities (optional)
    start_probs = np.exp(start_logits - np.max(start_logits))
    start_probs /= np.sum(start_probs)
    end_probs = np.exp(end_logits - np.max(end_logits))
    end_probs /= np.sum(end_probs)

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


def generate_answer(context, query):
    input_text = f"{context} \n\n {query}"
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=100, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

for example in dataset["test"]:
    context = example["passage"] + " \n\n " + example["query"] 
    query = "What is the value of @placeholder, based on the given text?"
    answer = generate_answer(context, query)
    print(context)
    print("\n\n\n\n\n")
    print(answer)