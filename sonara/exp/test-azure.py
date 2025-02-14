import azure.cognitiveservices.speech as speechsdk
import time
import os

def main():
    # 替换为你的 Azure Speech 订阅 Key 和区域
    speech_key = "b0e30eee618f423eb1f7bcf6dbdc2e94"
    service_region = "westus2"
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
    
    # 创建一个 Push Audio 流，用于推送音频数据
    stream = speechsdk.audio.PushAudioInputStream()
    # 通过推送流创建 AudioConfig
    audio_config = speechsdk.audio.AudioConfig(stream=stream)
    
    # 使用语音配置和音频配置创建 SpeechRecognizer
    recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
    
    # 注册实时识别事件
    recognizer.recognizing.connect(lambda evt: print("识别中: {}".format(evt.result.text)))
    recognizer.recognized.connect(lambda evt: print("识别结果: {}".format(evt.result.text)))
    recognizer.canceled.connect(lambda evt: print("识别取消: {}".format(evt)))
    recognizer.session_started.connect(lambda evt: print("会话开始"))
    recognizer.session_stopped.connect(lambda evt: print("会话结束"))
    
    # 启动连续语音识别
    recognizer.start_continuous_recognition()
    
    # 指定存储 chunk 文件的目录，替换为你实际的目录路径
    # chunk_dir = "/tmp/audio-4e6f4960cdf6455f8b0749b9362b4ef6"
    chunk_dir = "/tmp/audio-e57888d4a70e4a76b2ccb9cd1ffe7af2"
    # 获取目录下所有 .webm 文件，按文件名排序（假定文件名按录制顺序命名）
    chunk_files = sorted([f for f in os.listdir(chunk_dir) if f.endswith('.webm')])
    
    print("即将发送 {} 个 chunk 文件...".format(len(chunk_files)))
    
    for chunk_file in chunk_files:
        file_path = os.path.join(chunk_dir, chunk_file)
        print("发送 chunk 文件: {}".format(file_path))
        with open(file_path, 'rb') as f:
            data = f.read()
            stream.write(data)
        # 模拟实时录制：每发送一个 chunk 后等待1秒
        time.sleep(1)
    
    # 所有数据发送完毕后关闭推送流
    stream.close()
    
    # 等待一段时间以确保 Azure 完成识别处理（可根据需要调整时间）
    time.sleep(5)
    
    # 停止连续识别
    recognizer.stop_continuous_recognition()

if __name__ == '__main__':
    main()
