from transformers import AutoTokenizer, GPTNeoXForQuestionAnswering
import torch
import json

def load_data(filename):
    """Load JSON data from a file."""
    with open(filename, 'r') as file:
        data = json.load(file)
    return data['data']


def main():
    data = load_data('dev.json')
    
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    model = GPTNeoXForQuestionAnswering.from_pretrained("EleutherAI/gpt-neox-20b")
    

    for item in data:
        passage = item['passage']['text']
        for qa in item['qas']:
            question = qa['query']
            text = passage

            inputs = tokenizer(question, text, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)

            answer_start_index = outputs.start_logits.argmax()
            answer_end_index = outputs.end_logits.argmax()

            # Decode the predicted answer
            answer_start = torch.argmax(answer_start_scores)  # Get the most likely beginning of answer with the max score
            answer_end = torch.argmax(answer_end_scores) + 1  # Get the most likely end of answer, add 1 because end index is exclusive
            answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end]))

            print(answer)

if __name__ == "__main__":
    main()