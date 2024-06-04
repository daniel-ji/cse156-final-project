import csv
from sklearn.metrics import f1_score

def load_data(filename):
    predictions, actuals = [], []
    with open(filename, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter='\t')
        for row in reader:
            if len(row) == 2:  # Ensure there are exactly two columns
                predictions.append(row[0])
                actuals.append(row[1])
            else: 
                raise ValueError("Each row must contain exactly two columns.")

    return predictions, actuals

def calculate_metrics(predictions, actuals):
    f1 = f1_score(actuals, predictions, average='weighted')

    # Exact match calculation
    exact_matches = sum(1 for i in range(len(predictions)) if predictions[i] == actuals[i])
    exact_match_score = exact_matches / len(predictions)

    return f1, exact_match_score

def main():
    predictions, actuals = load_data('raw_predictions.tsv')
    f1, exact_match_score = calculate_metrics(predictions, actuals)
    
    print("F1 Score:", f1)
    print("Exact Match Score:", exact_match_score)

if __name__ == "__main__":
    main()
