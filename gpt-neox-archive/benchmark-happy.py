from happytransformer import HappyGeneration
from datasets import load_dataset
import torch
import numpy as np

happy_gen = HappyGeneration("GPT-NEO", "EleutherAI/gpt-neox-20b")

dataset = load_dataset("aps/super_glue", "record")

device = "cuda" if torch.cuda.is_available() else "cpu"

for example in dataset["test"]:
    query = example["passage"] + "\n\n" + example["query"] + "\n\nWhat is the value of @placeholder, based on the given text?"
    answer = happy_gen.generate_text(query)
    print("Query: ", query)
    print("\n\n\n\n\n")
    print(answer)
    print("\n\n\n\n\n")