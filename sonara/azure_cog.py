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
        speech_config.speech_recognition_language = "en-US"  # 或者其他语言设置
        speech_config.enable_automatic_punctuation = True

        # 启用说话人分离中间结果
        speech_config.set_property(
            property_id=speechsdk.PropertyId.SpeechServiceResponse_DiarizeIntermediateResults,
            value='true'
        )

        # create a push audio stream and create an AudioConfig based on it
        self.push_stream = speechsdk.audio.PushAudioInputStream()
        audio_config = speechsdk.audio.AudioConfig(stream=self.push_stream)
        
        # 创建会话转录器（而不是语音识别器）
        self.conversation_transcriber = speechsdk.transcription.ConversationTranscriber(
            speech_config=speech_config,
            audio_config=audio_config
        )
        
        # 注册会话转录事件
        self.conversation_transcriber.transcribing.connect(self.handle_transcribing)
        self.conversation_transcriber.transcribed.connect(self.handle_transcribed)  # 确保此方法存在
        self.conversation_transcriber.canceled.connect(lambda evt: print(f"转录取消: {evt}"))
        self.conversation_transcriber.session_started.connect(lambda evt: print("会话开始"))
        self.conversation_transcriber.session_stopped.connect(lambda evt: print("会话结束"))
        
        # 开始连续转录
        self.conversation_transcriber.start_transcribing_async()
        print("Azure conversation transcriber started")
    
    def handle_transcribing(self, evt):
        """实时转录回调"""
        if evt.result.text:
            # 获取说话人ID和说话人信息
            speaker_id = "unknown"
            if hasattr(evt.result, "speaker_id"):
                speaker_id = evt.result.speaker_id
            elif hasattr(evt.result, "speaker") and evt.result.speaker:
                speaker_id = evt.result.speaker
            
            message = json.dumps({
                "type": "recognizing", 
                "result": evt.result.text,
                "speaker": speaker_id
            })
            asyncio.run_coroutine_threadsafe(self.websocket.send(message), self.loop)
            print(f"发送实时转录结果: {evt.result.text}, 说话人: {speaker_id}")
        
    def handle_transcribed(self, evt):
        """最终转录回调"""
        if not evt.result.text:
            return
        
        # 获取说话人ID
        speaker_id = "unknown"
        if hasattr(evt.result, "speaker_id"):
            speaker_id = evt.result.speaker_id
        elif hasattr(evt.result, "speaker") and evt.result.speaker:
            speaker_id = evt.result.speaker
        
        message = json.dumps({
            "type": "recognized", 
            "result": evt.result.text,
            "speaker": speaker_id
        })
        asyncio.run_coroutine_threadsafe(self.websocket.send(message), self.loop)
        print(f"发送最终转录结果: {evt.result.text}, 说话人: {speaker_id}")
        
        # 传递说话人ID给翻译函数
        asyncio.run_coroutine_threadsafe(self.call_translation(evt.result.text, speaker_id), self.loop)
        
    async def call_translation(self, text: str, speaker_id="unknown"):
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
                translated_message = json.dumps({
                    "type": "translated", 
                    "result": translation,
                    "speaker": speaker_id
                })
                await self.websocket.send(translated_message)
                print(f"发送翻译结果: {translation}, 说话人: {speaker_id}")
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
        self.conversation_transcriber.stop_transcribing_async()  # 使用正确的方法和对象
        print("Azure speech recognizer stopped")
