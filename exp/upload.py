import os
import time
import threading
import tempfile
from pydub import AudioSegment
import azure.cognitiveservices.speech as speechsdk

# 请替换为你的 Azure Speech 服务订阅信息
speech_key = "7ca159be1fa34580a8e5a51a6381dd25"
service_region = "westus"

def recognizing_handler(evt):
    # 输出临时识别结果
    print("Recognizing: {}".format(evt.result.text))

def recognized_handler(evt):
    # 输出最终识别结果
    print("Recognized: {}".format(evt.result.text))

def process_segment(segment_wav_path, segment_index):
    print(f"\n=== Processing segment {segment_index} ({segment_wav_path}) ===")
    
    # 创建音频输入配置（传入当前段 WAV 文件）
    audio_config = speechsdk.audio.AudioConfig(filename=segment_wav_path)
    # 创建语音识别配置
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
    # 根据需要设置识别语言，这里以中文为例
    speech_config.speech_recognition_language = "zh-CN"
    
    # 创建语音识别器
    recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
    
    # 订阅临时识别事件
    recognizer.recognizing.connect(recognizing_handler)
    # 订阅最终识别事件
    recognizer.recognized.connect(recognized_handler)
    
    # 通过 session_stopped 和 canceled 事件来判断识别何时结束
    done_event = threading.Event()
    def stop_cb(evt):
        done_event.set()
    recognizer.session_stopped.connect(stop_cb)
    recognizer.canceled.connect(stop_cb)
    
    # 开始连续识别（连续识别模式下可获得 recognizing 事件）
    recognizer.start_continuous_recognition()
    
    # 等待识别结束（1秒的音频一般不会太久，这里设置超时 3 秒）
    done_event.wait(timeout=3)
    
    recognizer.stop_continuous_recognition()

def main():
    opus_file = "demo.opus"
    
    # 使用 pydub 载入 opus 文件，注意指定 format 为 "opus"
    audio = AudioSegment.from_file(opus_file, format="opus")
    print(f"原始音频时长：{len(audio)/1000:.2f}秒, 通道数：{audio.channels}, 采样率：{audio.frame_rate}")
    
    # 转换为 Azure 推荐格式：16kHz、单声道、16位 PCM
    audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
    
    duration_ms = len(audio)
    print(f"转换后音频时长：{duration_ms/1000:.2f}秒")
    
    # 按 1 秒（1000 毫秒）切分音频
    segment_duration_ms = 1000
    num_segments = (duration_ms + segment_duration_ms - 1) // segment_duration_ms
    print(f"将音频切分成 {num_segments} 个 1 秒左右的片段")
    
    for i in range(0, duration_ms, segment_duration_ms):
        segment = audio[i:i+segment_duration_ms]
        segment_index = i // segment_duration_ms
        
        # 将每个片段写入临时 WAV 文件
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
            segment.export(tmp_wav.name, format="wav")
            tmp_filename = tmp_wav.name
        
        try:
            process_segment(tmp_filename, segment_index)
        finally:
            # 处理完后删除临时文件
            os.remove(tmp_filename)

if __name__ == "__main__":
    main()
