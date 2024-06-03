from transformers import GPTNeoXTokenizerFast, GPTNeoXForQuestionAnswering
import torch
import json
import numpy as np

def load_data(filename):
    """Load JSON data from a file."""
    with open(filename, 'r') as file:
        data = json.load(file)
    return data['data']

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

def main():
    data = load_data('dev.json')

    tokenizer = GPTNeoXTokenizerFast.from_pretrained("EleutherAI/gpt-neox-20b")
    model = GPTNeoXForQuestionAnswering.from_pretrained("EleutherAI/gpt-neox-20b", torch_dtype=torch.float16).cuda()

    max_tests = 1000
    curr_tests = 0

    file_data = []

    for item in data:
        passage = item['passage']['text']
        if curr_tests >= max_tests:
            break
        for qa in item['qas']:
            question = "Read the text and answer the question at the end of the text."
            adapted_passage = passage + "\n\nWhat is the value of @placeholder, based on the given text? " + qa['query']

            inputs = tokenizer(question, adapted_passage, return_tensors="pt")
            # Move the tensor to CUDA
            inputs = {k: v.cuda() for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)

            # Get the logits from the outputs
            answer_start_index, answer_end_index = find_best_answer_span(outputs.start_logits[0].cpu().numpy(), outputs.end_logits[0].cpu().numpy())

            # Convert tokens to string
            answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start_index:answer_end_index]))

            file_data.append({
                "question": question,
                "passage": adapted_passage,
                "answer": answer,
                "answer_start_index": answer_start_index,
                "answer_end_index": answer_end_index,
            })

        curr_tests += 1
            
    with open('results.txt', 'w') as f:
        for item in file_data:
            f.write(f"Question: {item['question']}\n")
            f.write(f"Passage: {item['passage']}\n")
            f.write(f"Answer: {item['answer']}\n")
            f.write(f"Answer Start Index: {item['answer_start_index']}\n")
            f.write(f"Answer End Index: {item['answer_end_index']}\n")


if __name__ == "__main__":
    main()
