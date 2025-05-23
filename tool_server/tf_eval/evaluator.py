import argparse
import json
import re
from math_verify import parse, verify
from thefuzz import fuzz


def evaluate(ground_truth_path, processed_path):
    ground_truth = dict()
    with open(ground_truth_path, 'r', encoding='utf-8') as f:
        all_data = json.load(f)
    if 'chartgemma' in ground_truth_path:
        for data in all_data:
            ground_truth[data['question']] = data['label'].replace('<answer> ', '').replace(' </answer>', '')

    all_data = []
    with open(processed_path, 'r', encoding='utf-8') as f:
        for line in f:
            all_data.append(json.loads(line))

    processed_data = dict()
    pattern_list = [
        r'"actions": \[\{"name": "Terminate", "arguments": \{"ans": (.*?)\}',
        r'"actions": \[\{"name": "Terminate", "arguments": \{"answer": (.*?)\}'
    ]

    for data in all_data:
        try:
            model_response = data['results']['results']['conversation'][-1]['content'][0]['text']
            final_action = "{\"actions\": " + model_response.split("\"actions\": ")[1]
            for pattern in pattern_list:
                matches = re.findall(pattern, final_action)
                if len(matches) == 1:
                    pred = matches[0].replace(r'"', '')
                    key = data['results']['results']['meta_data']['text']
                    processed_data[key] = pred
                    break
        except Exception:
            pass

    acc = 0
    for item in processed_data.items():
        if item[0] not in ground_truth:
            print('error')
            continue
        gold = parse('${0}$'.format(ground_truth[item[0]]))
        pred = parse('${0}$'.format(item[1]))
        if verify(gold, pred):
            acc += 1
        elif '%' in item[1][-1]:
            pred = item[1][:-1]
            if '.' in pred:
                pred = pred.split('.')[0]
            if pred == ground_truth[item[0]]:
                acc += 1
        elif '.' in item[1]:
            pred = item[1].split('.')[0]
            if pred == ground_truth[item[0]]:
                acc += 1
        else:
            acc += fuzz.ratio(ground_truth[item[0]], item[1]) / 100

    print('processed path: {0}, acc: {1:.2f}'.format(processed_path, acc / len(all_data) * 100))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model outputs against ground truth.")
    parser.add_argument("--ground_truth", type=str, required=True, help="Path to ground truth JSON file.")
    parser.add_argument("--processed", type=str, required=True, help="Path to model processed output JSON file.")

    args = parser.parse_args()

    evaluate(args.ground_truth, args.processed)
