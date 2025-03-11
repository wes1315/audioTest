import os
import json
import asyncio
import azure.cognitiveservices.speech as speechsdk
from sonara.llm_translate import translate_text_with_retries
from sonara.groq_translator import GroqTranslator


class AzureCognitiveService:

    """
    wrap the azure continuous recognition logic, and send two types of messages through the websocket during the recognition process:
      - "recognizing": during the real-time recognition
      - "recognized": when a whole sentence is recognized
    """
    def __init__(self, websocket, loop):
        """
        :param websocket: the current websocket connection
        :param loop: asyncio event loop, for scheduling websocket send in the callback thread
        """
        self.websocket = websocket
        self.loop = loop

        self.groq_translator = GroqTranslator(
            api_key=os.getenv("GROQ_API_KEY"),
            model=os.getenv("GROQ_MODEL")
        )

        # load azure config from environment variables
        speech_key = os.getenv("AZURE_SUBSCRIPTION_KEY")
        service_region = os.getenv("AZURE_REGION")
        if not speech_key or not service_region:
            raise ValueError("please ensure AZURE_SUBSCRIPTION_KEY and AZURE_REGION are set in environment variables")
        
        speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
        # if you need to set the recognition language, for example, chinese, you can enable the following line:
        # speech_config.speech_recognition_language = "zh-CN"
        
        # create a push audio stream and create an AudioConfig based on it
        self.push_stream = speechsdk.audio.PushAudioInputStream()
        audio_config = speechsdk.audio.AudioConfig(stream=self.push_stream)
        
        # create a speech recognizer (continuous recognition)
        self.recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
        
        # register continuous recognition events
        self.recognizer.recognizing.connect(self.handle_recognizing)
        self.recognizer.recognized.connect(self.handle_recognized)
        self.recognizer.canceled.connect(lambda evt: print("recognizer canceled: {}".format(evt)))
        self.recognizer.session_started.connect(lambda evt: print("session started"))
        self.recognizer.session_stopped.connect(lambda evt: print("session stopped"))
        
        self.recognizer.start_continuous_recognition()
        print("Azure speech recognizer started")
    
    def handle_recognizing(self, evt):
        """
        callback for recognizing
        """
        text = evt.result.text
        if text:
            message = json.dumps({"type": "recognizing", "result": text})
            # in the callback thread of azure, call websocket.send by run_coroutine_threadsafe
            asyncio.run_coroutine_threadsafe(self.websocket.send(message), self.loop)
            print("sending recognizing result: {}".format(text))
    
    def handle_recognized(self, evt):
        """
        callback for recognized
        """
        text = evt.result.text
        if not text:
            return
        message = json.dumps({"type": "recognized", "result": text})
        asyncio.run_coroutine_threadsafe(self.websocket.send(message), self.loop)
        print("sending final recognized result: {}".format(text))
        asyncio.run_coroutine_threadsafe(self.call_translation(text), self.loop)

    async def call_translation(self, text: str):
        """
        调用异步翻译函数，并将翻译结果通过 websocket 发送给前端
        """
        try:
            translation = await self.loop.run_in_executor(
                None, 
                self.groq_translator.translate_with_retries, 
                text
            )
            if translation:
                translated_message = json.dumps({"type": "translated", "result": translation})
                await self.websocket.send(translated_message)
                print("sending translated result: {}".format(translation))
        except Exception as e:
            print("Translation failed:", e)

    def write(self, data: bytes):
        """
        write the received audio data to the push stream
        """
        self.push_stream.write(data)
    
    def close(self):
        """
        close the push stream and stop the recognizer
        """
        self.push_stream.close()
        self.recognizer.stop_continuous_recognition()
        print("Azure speech recognizer stopped")
