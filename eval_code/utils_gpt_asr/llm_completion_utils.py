from openai import OpenAI
import time

# for gpt
def chatCompletion(model, messages, temperature, retry_times, round_sleep, fail_sleep, api_key, base_url=None):
    if base_url is None:
        client = OpenAI(
            api_key=api_key
            )
    else:
        client = OpenAI(
        api_key=api_key,
        base_url=base_url
    )
    try:
        response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature
            )
    except Exception as e:
        print(e)
        for retry_time in range(retry_times):
            retry_time = retry_time + 1
            print(f"{model} Retry {retry_time}")
            time.sleep(fail_sleep)
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature
                )
                break
            except:
                continue

    model_output = response.choices[0].message.content.strip()
    time.sleep(round_sleep)

    return model_output