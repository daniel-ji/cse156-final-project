import json

def load_data(filename):
    """Load JSON data from a file."""
    with open(filename, 'r') as file:
        data = json.load(file)
    return data['data']

def calculate_text_statistics(texts):
    """Calculate statistics for a list of text passages."""
    total_characters = sum(len(text) for text in texts)
    total_words = sum(len(text.split()) for text in texts)
    num_texts = len(texts)
    average_length = total_characters / num_texts if num_texts else 0
    average_word_count = total_words / num_texts if num_texts else 0
    
    return {
        "total_characters": total_characters,
        "total_words": total_words,
        "average_length": average_length,
        "average_word_count": average_word_count
    }

def main():
    data = load_data('dev.json')
    
    # Extract passages and queries
    passages = [item['passage']['text'] for item in data]
    queries = [qa['query'] for item in data for qa in item['qas']]
    
    # Calculate statistics
    passage_stats = calculate_text_statistics(passages)
    query_stats = calculate_text_statistics(queries)
    
    print("Passage Statistics:")
    for key, value in passage_stats.items():
        print(f"{key}: {value}")

    print("\nQuery Statistics:")
    for key, value in query_stats.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()
