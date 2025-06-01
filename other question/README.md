# Other issue

## chat template

As we mentioned in Appendix A.3.2, *Previous jailbreak attempts typically used the fastchat package to manage context. However, the settings of the new models do not synchronize with the package in a timely manner. Therefore, in this paper, we set all the prompts for our experiments (including system prompts, etc.) using the templates provided by the model provider in the “tokenizer\_config.json” file, in conjunction with Hugging Face’s “apply\_chat\_template” function. For the baseline methods, we made cor*
*responding adaptations to ensure that the templates remained consistent.*

Therefore, we have adapted the chat templates used by SIJ. It is particularly important to note that for the Llama2-7b-chat model, its default chat template does not include a system prompt. However, in some baseline methods, such as SafeDecoding, a system prompt is adapted, which is a key point to be aware of.

If you also need to adapt the fschat package, you can use the adaptation function we wrote. A simple example is as follows:

```python
from .fschat_transformer_adapter import adapter

def load_conversation_template(template_name):
    conv_template = adapter(template_name)
    return conv_template
```

## experiment result

As a reminder, the “prompt” field in our experimental results file contains **all the content to be inputted to the model, not just the user prompt.** If you want to directly reproduce the results, please do not apply any other chat template to it, or extract only the user prompt from it and then use the `tokenizer.apply_chat_template()` function.
