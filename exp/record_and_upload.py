# record_and_upload.py
import os
import uuid
from dotenv import load_dotenv
load_dotenv()  # 自动加载当前目录下的 .env 文件

import pyaudio
import wave
import threading
import azure.cognitiveservices.speech as speechsdk

# 从环境变量中读取 Azure 配置
AZURE_SUBSCRIPTION_KEY = os.getenv("AZURE_SUBSCRIPTION_KEY")
AZURE_REGION = os.getenv("AZURE_REGION")
print(f"AZURE_SUBSCRIPTION_KEY: {AZURE_SUBSCRIPTION_KEY}")
print(f"AZURE_REGION: {AZURE_REGION}")

if not AZURE_SUBSCRIPTION_KEY or not AZURE_REGION:
    raise ValueError("请确保 .env 文件中设置了 AZURE_SUBSCRIPTION_KEY 和 AZURE_REGION")


# 自动生成保存录音文件的目录：/tmp/wav-{hashcode}
hashcode = uuid.uuid4().hex[:8]
AUDIO_DIR = f"/tmp/wav-{hashcode}"
os.makedirs(AUDIO_DIR, exist_ok=True)
print(f"音频文件将保存到目录：{AUDIO_DIR}")

def recognize_audio(file_path):
    """
    使用 Azure Speech SDK 对指定的音频文件进行一次性识别
    """
    speech_config = speechsdk.SpeechConfig(subscription=AZURE_SUBSCRIPTION_KEY, region=AZURE_REGION)
    audio_config = speechsdk.audio.AudioConfig(filename=file_path)
    recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    result = recognizer.recognize_once()

    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        print(f"[识别成功] 文件 {file_path} 的识别结果：{result.text}")
    else:
        print(f"[识别失败] 文件 {file_path} 识别失败，原因：{result.reason}")

def record_and_upload():
    """
    实时录音，每1秒保存一个音频文件到 AUDIO_DIR，并异步上传进行识别
    """
    # 录音参数配置
    CHUNK = 1024               # 每个数据块的帧数
    FORMAT = pyaudio.paInt16   # 采样位数
    CHANNELS = 1               # 单声道
    RATE = 16000               # 采样率 (Hz)
    RECORD_SECONDS = 1         # 每段录音时长（秒）

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("开始录音，按 Ctrl+C 停止...")

    index = 0
    try:
        while True:
            frames = []
            num_chunks = int(RATE / CHUNK * RECORD_SECONDS)
            for i in range(num_chunks):
                data = stream.read(CHUNK)
                frames.append(data)

            # 构造文件完整路径，例如 /tmp/wav-{hashcode}/audio_0.wav, /tmp/wav-{hashcode}/audio_1.wav, ...
            filename = os.path.join(AUDIO_DIR, f"audio_{index:05d}.wav")
            wf = wave.open(filename, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
            wf.close()

            print(f"已保存文件：{filename}")

            # 异步线程进行识别（不阻塞录音过程）
            threading.Thread(target=recognize_audio, args=(filename,)).start()
            index += 1
    except KeyboardInterrupt:
        print("录音结束。")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == '__main__':
    record_and_upload()
