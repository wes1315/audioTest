from logging import debug, info
from ollama import chat, ChatResponse
import re

def translate_text(text: str) -> str:
    prompt = f"""
Please help me translate the following paragraph into Chinese and wrap it with <START> and <END>:

<START>
{text}
<END>
"""
    debug("prompt: {}".format(prompt))
    response: ChatResponse = chat(model='llama3.2:3b', messages=[
        {'role': 'user', 'content': prompt}
    ])

    debug("response: {}".format(response))
    content = response['message']['content']
    match = re.search(r"<START>(.*?)<END>", content, re.DOTALL)
    debug("content: {}".format(content))
    if match:
        translation = match.group(1).strip()
        debug("translation: {}".format(translation))
        return translation
    
    debug("no match")
    return ""


def translate_text_with_retries(text: str, retries: int = 3) -> str:
    for _ in range(retries):
        translation = translate_text(text)
        if translation:
            return translation
    return ""
