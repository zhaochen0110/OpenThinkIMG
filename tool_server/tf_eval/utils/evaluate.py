from math_verify import parse, verify
from typing import List

def evaluate_metric(ground_truth: List[str], prediction_data: List[str]):
    for label, prediction in zip(ground_truth, prediction_data):
        gold = parse('${0}$'.format(label))
        pred = parse('${0}$'.format(prediction))
        if verify(gold, pred):
            acc+=1
        elif '%' in prediction[-1]:
            pred = prediction[:-1:]
            if '.' in pred:
                pred = pred.split('.')[0]
            if pred == ground_truth[item[0]]:
                acc+=1
        elif '.' in prediction:
            pred = prediction.split('.')[0]
            if pred == ground_truth[item[0]]:
                acc+=1
    return acc/len(ground_truth)