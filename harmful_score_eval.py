# Evaluation
import os
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from safe_eval import GPTJudge
import json, logging
import argparse

def evaluate_harmful_score(input_name, api, baseurl):

    with open("dataset/harmful_behaviors_custom", 'r') as file_baseline:
        baseline_s = json.load(file_baseline)

    with open(input_name, 'r') as file:
        output_json = json.load(file)

    logging.info("Evaluating...")
    results = output_json
    goals = [baseline['goal'] for baseline in baseline_s]
    instructions = ["same as before" for result in results]
    responses = [result['output'] for result in results]

    gpt_judge = GPTJudge('gpt',mp=20, api=api, baseurl=baseurl)
    goals_responses_pairs = []
    for i in range(len(instructions)):
        goals_responses_pairs.append((goals[i], responses[i]))

    scores, reasons = gpt_judge.evaluate(goals_responses_pairs)
    scores_gpt = gpt_judge.score_parse(scores)

    safe_eval_results = []

    for i in range(len(instructions)):
        output_formatted = {
            "id": i,
            "goal": goals[i],
            "instruction": instructions[i],
            "output": responses[i],
            "GPT_score": scores_gpt[i],
            "GPT_reason": reasons[i]
        }
        safe_eval_results.append(output_formatted)

    gpt_score_sum = 0
    defense_success_count = 0
    for i in range(len(safe_eval_results)):
        if safe_eval_results[i]['GPT_score'] != None:
            gpt_score_sum += safe_eval_results[i]['GPT_score']

    print(f'Harmful Score: {gpt_score_sum / len(safe_eval_results)}', flush=True)

    output_name = input_name.replace(".json", "")
    output_name = output_name + "_harmful_score.json"

    directory = os.path.dirname(output_name)

    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(output_name, 'w') as f:
        json.dump(safe_eval_results, f, indent=4)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_name", type=str, default="")
    parser.add_argument("--api", type=str, default="")
    parser.add_argument("--baseurl", type=str, default="")
    return parser.parse_args()

args = get_args()

evaluate_harmful_score(args.input_name, args.api, args.baseurl)