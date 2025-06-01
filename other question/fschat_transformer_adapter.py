# This file is for fschat==0.2.36 and 7B version of LLM

from fastchat import model as model_conversation
def adapter(path):
    conversation = model_conversation.get_conversation_template(path)
    # Llama-2-7b-chat-hf
    if "Llama-2-7b-chat-hf" in path:
        conversation.name = "Llama-2-chat-hf"
        return conversation

    # vicuna-v1.5
    if "vicuna-7b-v1.5" in path:
        conversation.name = "vicuna-v1.5"
        conversation.sep2 = "</s> "
        return conversation
    
    # Mistral-Instruct-v0.3
    if "Mistral-7B-Instruct-v0.3" in path:
        conversation.name = "Mistral-Instruct-v0.3"
        conversation.sep = ""
        def get_prompt(self) -> str:
            system_prompt = self.system_template.format(system_message=self.system_message)
            seps = [self.sep, self.sep2]
            if self.system_message:
                ret = system_prompt
            else:
                ret = "[INST] "
            for i, (role, message) in enumerate(self.messages):
                tag = self.roles[i % 2]
                if message:
                    if i == 0:
                        # delete the " " that append after ret
                        ret += message
                    else:
                        ret += tag + " " + message + seps[i % 2]
                else:
                    ret += tag
            return ret
        conversation.get_prompt = get_prompt.__get__(conversation, conversation.__class__)
        return conversation
    
    # Llama-3.1-8B-Instruct
    if "Llama-3.1-8B-Instruct" in path:
        conversation.name = "Llama-3.1-Instruct"

        # clear messages
        conversation.messages = []
        conversation.system_template = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_message}\n\n<|eot_id|>"
        conversation.system_message = "Cutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024"
        conversation.roles = ("<|start_header_id|>user<|end_header_id|>", "<|start_header_id|>assistant<|end_header_id|>")
        conversation.sep = "<|eot_id|>"
        def get_prompt(self) -> str:
            system_prompt = self.system_template.format(system_message=self.system_message)
            sep = self.sep
            if self.system_message:
                ret = system_prompt
            for i, (role, message) in enumerate(self.messages):
                tag = self.roles[i % 2]
                if message:
                        ret += tag + "\n\n" + message + sep
                else:
                    ret += tag + "\n\n"
            return ret
        conversation.get_prompt = get_prompt.__get__(conversation, conversation.__class__)
        return conversation

    # deepseek-llm-7b-chat
    if "deepseek-llm-7b-chat" in path:
        conversation.name = "deepseek-llm-chat"
        return conversation