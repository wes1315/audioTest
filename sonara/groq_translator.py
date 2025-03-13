import os
from groq import Groq


class GroqTranslator:
    def __init__(self, api_key, model: str = "llama-3.3-70b-versatile"):
        """
        Initialize the GroqTranslator class, create a Groq client.
        :param model: The name of the model to use, default is "llama-3.3-70b-versatile"
        """
        self.client = Groq(api_key=api_key)
        self.model = model

    def translate(self, text: str) -> str:
        """
        Call the Groq conversation interface to translate the given text and return the translation result.
        The translation content is between the <START> and <END> tags in the returned result.
        :param text: The text string to be translated
        :return: The translated text
        """
        if not text or text.strip() == "":
            print("Warning: Empty text provided for translation")
            return ""

        print(f"Translating text: '{text}'")
        prompt = f"""
请帮我翻译下面<START>/<END>中间的内容，然后把翻译结果也放在response的<START>/<END>中间.

<START>
{text}
<END>
"""
        try:
            # Call the Groq conversation interface
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model=self.model,
            )

            # Get the complete response text from the model
            response_text = chat_completion.choices[0].message.content
            print(f"Raw translation response: '{response_text}'")

            # Parse the translation content between <START> and <END>
            start_tag = "<START>"
            end_tag = "<END>"
            start_idx = response_text.find(start_tag)
            end_idx = response_text.find(end_tag)

            if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
                translation = response_text[start_idx + len(start_tag):end_idx].strip()
                print(f"Extracted translation: '{translation}'")
                return translation
            else:
                print(f"Warning: Could not find START/END tags in response: '{response_text}'")
                return response_text
        except Exception as e:
            print(f"Error in translation: {str(e)}")
            return f"Translation error: {str(e)}"

    def translate_with_retries(self, text: str, retries: int = 3) -> str:
        """
        Try to translate with retries in case of failures
        """
        last_error = None
        for attempt in range(retries):
            try:
                print(f"Translation attempt {attempt+1}/{retries}")
                translation = self.translate(text)
                if translation:
                    return translation
                print(f"Empty translation result on attempt {attempt+1}")
            except Exception as e:
                last_error = e
                print(f"Translation attempt {attempt+1} failed: {str(e)}")
        
        print(f"All {retries} translation attempts failed")
        if last_error:
            return f"Translation failed after {retries} attempts: {str(last_error)}"
        return "Translation failed with no specific error"
