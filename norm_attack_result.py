import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import argparse

def eval(model_name, model_path, prompt_path, method_name):
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    result = []
    with open(prompt_path, 'r', encoding='utf-8') as file:
        attack_prompts = json.load(file)

    if method_name == "sij":
        time_all = 0
        for prompt in attack_prompts[1:]:
            time_all += prompt["run time"]
            temp = {}
            temp["id"] = prompt["id"]
            temp["goal"] = prompt["goal"]
            temp["prompt"] = prompt["prompt"]
            if model_name == "vicuna":
                a = tokenizer(temp["prompt"] , return_tensors="pt").input_ids.to(model.device)
                response = tokenizer.batch_decode(model.generate(input_ids=a, do_sample=False, max_new_tokens=256))[0]
                temp["response"] = response[len(temp["prompt"]):]
                print(temp["response"], flush=True)
            else:
                # delete start token
                a = tokenizer(temp["prompt"], return_tensors="pt").input_ids[..., 1:].to(model.device)
                response = tokenizer.batch_decode(model.generate(input_ids=a, do_sample=False, max_new_tokens=256))[0]
                temp["response"] = response[len(temp["prompt"]):]
                print(temp["response"], flush=True)
            result.append(temp)

        average_time = time_all/50
        result.append({"time": average_time})
    
    elif method_name == "gcg":
        if model_name != "llama2":
            start_labels = []
            time_all = 0
            label = attack_prompts["losses"]

            i = 0
            while i <len(label):
                if attack_prompts["losses"][i] == 1000000.0 and attack_prompts["losses"][i+1] == 1000000.0:
                    start_labels.append(i)
                    i += 2
                    continue
                i += 1

            print(len(start_labels))
            if len(start_labels) != 50:
                raise ValueError("The value should be 50.")

            for i in range(len(label)):
                time_all += attack_prompts["runtimes"][i]

            for i in range(50):
                temp = {}
                temp["id"] = i
                temp["goal"] = attack_prompts["params"]["goals"][i]
                if i !=49:
                    suffix = attack_prompts["controls"][start_labels[i+1]-1]
                if i == 49:
                    suffix = attack_prompts["controls"][-1]
                temp["prompt"] = temp["goal"] + " " + suffix
                chat = [
                    {"role": "user", "content": temp["prompt"]}
                ]
                a = tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt").to(model.device)
                b = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
                c = tokenizer(b, return_tensors="pt").input_ids.to(model.device)
                # print(c)
                # raise ValueError("The value should be 25.")
                if model_name == "deepseek":
                    c = c[..., 0:-2]
                    # corresponding to the result of GCG
                response = tokenizer.batch_decode(model.generate(input_ids=c, do_sample=False, max_new_tokens=256))[0]
                temp["response"] = response[len(b):]
                print(temp["response"], flush=True)
                result.append(temp)
            average_time = time_all/50
            result.append({"time": average_time})

        elif model_name == "llama2":
            start_labels = []
            time_all = 0
            label = attack_prompts["losses"]
            for i in range(len(label)):
                if attack_prompts["losses"][i] == 1000000.0 and attack_prompts["losses"][i+1] == 1000000.0:
                    start_labels.append(i)
                    i += 1
            if len(start_labels) != 25:
                raise ValueError("The value should be 25.")

            for i in range(len(label)):
                time_all += attack_prompts["runtimes"][i]

            for i in range(25):
                temp = {}
                temp["id"] = i
                temp["goal"] = attack_prompts["params"]["goals"][i]
                if i != 24:
                    suffix = attack_prompts["controls"][start_labels[i+1]-1]
                if i == 24:
                    suffix = attack_prompts["controls"][-1]
                temp["prompt"] = temp["goal"] + " " + suffix
                chat = [
                    {"role": "user", "content": temp["prompt"]}
                ]
                a = tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt").to(model.device)
                b = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
                c = tokenizer(b, return_tensors="pt").input_ids.to(model.device)
                # <s> one start token corresponding to the result of GCG
                response = tokenizer.batch_decode(model.generate(input_ids=a.input_ids, do_sample=False, max_new_tokens=256))[0]
                # response = tokenizer.batch_decode(model.generate(**a, do_sample=False, max_new_tokens=256))[0]
                temp["response"] = response[len(b):]
                print(temp["response"], flush=True)
                result.append(temp)
            average_time = time_all/25
            result.append({"time": average_time})

    elif method_name == "renellm":
        time_all = 0
        for prompt in attack_prompts:
            time_all += prompt["time_cost"]
            temp = {}
            temp["id"] = prompt["idx"]
            temp["goal"] = prompt["original_harm_behavior"]
            temp["prompt"] = prompt["nested_prompt"]
            chat = [
                {"role": "user", "content": temp["prompt"]}
            ]
            a = tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt").to(model.device)
            b = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            response = tokenizer.batch_decode(model.generate(**a, do_sample=False, max_new_tokens=256))[0]
            temp["response"] = response[len(b):]
            result.append(temp)

        average_time = time_all/50
        result.append({"time": average_time})
    
    elif method_name == "deepinception":
        time_all = 0
        for prompt in attack_prompts:
            time_all += 0
            temp = {}
            temp["id"] = prompt["id"]
            temp["goal"] = prompt["goal"]
            temp["prompt"] = prompt["prompt"]
            chat = [
                {"role": "user", "content": temp["prompt"]}
            ]
            a = tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt").to(model.device)
            b = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            response = tokenizer.batch_decode(model.generate(**a, do_sample=False, max_new_tokens=256))[0]
            temp["response"] = response[len(b):]
            result.append(temp)

        average_time = time_all/50
        result.append({"time": average_time})
    
    elif method_name == "autodan":
        time_all = 0
        for i in range(50):
            time_all += attack_prompts[str(i)]["total_time"]
            temp = {}
            temp["id"] = i
            temp["goal"] = attack_prompts[str(i)]["goal"]
            temp["prompt"] = attack_prompts[str(i)]["final_suffix"].replace("[REPLACE]", temp["goal"].lower())
            chat = [
                {"role": "user", "content": temp["prompt"]}
            ]
            a = tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt").to(model.device)
            b = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            c = tokenizer(b, return_tensors="pt").input_ids.to(model.device)
            if model_name == "llama2":
                response = tokenizer.batch_decode(model.generate(input_ids=a.input_ids, do_sample=False, max_new_tokens=256))[0]
            else:
                response = tokenizer.batch_decode(model.generate(input_ids=c, do_sample=False, max_new_tokens=256))[0]
            temp["response"] = response[len(b):]
            print(temp["response"])
            result.append(temp)

        average_time = time_all/50
        result.append({"time": average_time})
    
    elif method_name =="none":
        time_all = 0
        for prompt in attack_prompts:
            time_all += 0
            temp = {}
            temp["id"] = prompt["id"]
            temp["goal"] = prompt["goal"]
            temp["prompt"] = prompt["goal"]
            chat = [
                {"role": "user", "content": temp["prompt"]}
            ]
            a = tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt").to(model.device)
            b = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            response = tokenizer.batch_decode(model.generate(**a, do_sample=False, max_new_tokens=256))[0]
            temp["response"] = response[len(b):]
            result.append(temp)

        average_time = time_all/50
        result.append({"time": average_time})
    
    save_path_dir = "all_norm_result/" + method_name + "/"
    if not os.path.exists(save_path_dir):
        os.makedirs(save_path_dir)
    file_name = model_name + ".json"
    # for self-reminder adapter attack
    file_name = model_name + "_SR_leak.json"

    with open(save_path_dir+file_name, 'w') as f:
        json.dump(result, f, indent=4)

def get_args():
    # Experiment Settings
    parser.add_argument("--model_name", type=str, default="vicuna")
    parser.add_argument("--prompt_path", type=str, default="")
    parser.add_argument("--method_name", type=str, default="")

    return parser.parse_args()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="norm")
    args = get_args()
    regester = {
        "vicuna":"vicuna-7b-v1.5",
        "llama2":"Llama-2-7b-chat-hf",
        "llama3":"Llama-3.1-8B-Instruct",
        "mistral":"Mistral-7B-Instruct-v0.3",
        "deepseek":"deepseek-llm-7b-chat",
    }
    eval(args.model_name, regester[args.model_name], args.prompt_path, args.method_name)
