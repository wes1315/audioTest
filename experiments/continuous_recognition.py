# continuous_recognition.py
import os
import sys
import time
import azure.cognitiveservices.speech as speechsdk
from dotenv import load_dotenv

def main():
    # 从 .env 文件中加载 Azure 配置
    load_dotenv()  
    speech_key = os.getenv("AZURE_SUBSCRIPTION_KEY")
    service_region = os.getenv("AZURE_REGION")
    if not speech_key or not service_region:
        raise ValueError("请确保 .env 文件中设置了 AZURE_SUBSCRIPTION_KEY 和 AZURE_REGION")
    
    # 创建语音配置（可根据需要设置识别语言，例如：zh-CN）
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
    # speech_config.speech_recognition_language = "zh-CN"  # 如需识别中文，请取消注释

    # 创建 Push Audio 流，并通过此流创建 AudioConfig
    push_stream = speechsdk.audio.PushAudioInputStream()
    audio_config = speechsdk.audio.AudioConfig(stream=push_stream)
    
    # 使用语音配置和音频配置创建 SpeechRecognizer（连续识别）
    recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
    
    # 注册连续识别的事件
    recognizer.recognizing.connect(lambda evt: print("识别中: {}".format(evt.result.text)))
    recognizer.recognized.connect(lambda evt: print("识别结果: {}".format(evt.result.text)))
    recognizer.canceled.connect(lambda evt: print("识别取消: {}".format(evt)))
    recognizer.session_started.connect(lambda evt: print("会话开始"))
    recognizer.session_stopped.connect(lambda evt: print("会话结束"))
    
    # 启动连续语音识别
    recognizer.start_continuous_recognition()
    
    # 通过命令行参数指定存放音频 chunk 的目录（例如：/tmp/wav-xxxxxx）
    if len(sys.argv) > 1:
        chunk_dir = sys.argv[1]
    else:
        print("请通过命令行参数指定音频 chunk 文件所在的目录，例如:")
        print("    python continuous_recognition.py /tmp/wav-xxxxxx")
        sys.exit(1)
    
    # 获取目录下所有 .wav 文件，并按文件名排序（假设文件命名为 audio_00000.wav、audio_00001.wav 等）
    chunk_files = sorted([f for f in os.listdir(chunk_dir) if f.endswith('.wav')])
    print("即将发送 {} 个 chunk 文件...".format(len(chunk_files)))
    
    # 按顺序读取每个文件，并将其数据写入推送流中
    for chunk_file in chunk_files:
        file_path = os.path.join(chunk_dir, chunk_file)
        print("发送 chunk 文件: {}".format(file_path))
        with open(file_path, 'rb') as f:
            data = f.read()
            push_stream.write(data)
        # 模拟实时录音：每发送一个 chunk 后等待 1 秒
        time.sleep(1)
    
    # 所有数据发送完毕后关闭推送流
    push_stream.close()
    
    # 等待一段时间，以确保 Azure 完成最后数据的识别（可根据需要调整等待时长）
    time.sleep(5)
    
    # 停止连续识别
    recognizer.stop_continuous_recognition()

if __name__ == '__main__':
    main()
