from transformers import AutoTokenizer, RobertaForMaskedLM
import torch
import json

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")
model = RobertaForMaskedLM.from_pretrained("FacebookAI/roberta-base").cuda()

# Load dataset
dataset = None
with open("dev.json", 'r') as file:
    dataset = json.load(file)['data']

file_data = []

count = 0
for sample in dataset:
    if not sample['qas'][0]["answers"] or len(sample['qas'][0]["answers"][0]) == 0:
        print("Skipping sample with no answers")
        continue

    # Ensure entities are considered for predictions
    entities_indexes = sample['passage']['entities']
    entities = []
    for index in entities_indexes:
        entities.append(sample['passage']['text'][index['start']:(index['end']+1)])

    # Mask the query
    masked_query = sample['qas'][0]['query'].replace("@placeholder", "<mask>")
    # Combine the passage and query
    query = sample['passage']['text'] + "\n\n" + masked_query

    with torch.no_grad():
        inputs = tokenizer(query, return_tensors="pt").to('cuda')
        # Should be only one sample in the dev.json file with more than 512 tokens
        if (len(inputs['input_ids'][0]) > 512):
            print("Skipping sample with too long query")
            continue

        # Get the logits from the model of the masked token
        outputs = model(**inputs)
        mask_token_index = (inputs['input_ids'] == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
        logits = outputs.logits[0, mask_token_index][0]

        # Get logits for each entity
        entity_ids = [tokenizer.encode(entity, add_special_tokens=False) for entity in entities]

        # Calculate likelihoods for each entity (average of the logits for each token in the entity)
        likelihoods = []
        for entity_id in entity_ids:
            likelihood = 0
            for token in entity_id:
                likelihood += logits[token]
            likelihoods.append(likelihood / len(entity_id))

        # Get the entity with the highest likelihood
        prediction = entities[likelihoods.index(max(likelihoods))]

        file_data.append((query, prediction, sample['qas'][0]["answers"][0]["text"]))

        # Write to file and log progress, every 100 samples
        if count % 100 == 0:
            print(f"Processed {count} samples")
            # Write the results to a file
            with open("predictions.txt", "a") as f:
                for query, prediction, answer in file_data:
                    f.write(f"Query: {query}\n")
                    f.write(f"Prediction: {prediction}\n")
                    f.write(f"Answer: {answer}\n")
                    f.write("\n")

            # Write raw results to a file (for further metric calculation)
            with open("raw_predictions.tsv", "a") as f:
                for _, prediction, answer in file_data:
                    f.write(f"{prediction}\t{answer}\n")

            file_data = []

        count += 1

