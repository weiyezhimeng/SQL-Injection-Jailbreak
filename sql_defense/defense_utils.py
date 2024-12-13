import random

register = {
    "vicuna":"vicuna-7b-v1.5",
    "llama2":"Llama-2-7b-chat-hf",
    "llama3":"Llama-3.1-8B-Instruct",
    "mistral":"Mistral-7B-Instruct-v0.3",
    "deepseek":"deepseek-llm-7b-chat",
}

def secret_generate(key, vocab_size):
    random.seed(key)
    random_numbers = [random.randint(0, vocab_size-1) for _ in range(5)]
    return random_numbers
