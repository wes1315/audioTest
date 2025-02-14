#!/usr/bin/env python3
import os
import wave

def rebuild_wav_using_wave(input_file, output_file, sample_rate=16000, channels=1, sampwidth=2):
    """
    读取 input_file 中的内容，跳过前 44 字节（原有 header），
    然后用 wave 模块重新包装成正确的 WAV 文件写入 output_file 中。
    
    参数说明：
        sample_rate: 采样率，默认 16000 Hz
        channels: 声道数，默认 1（单声道）
        sampwidth: 采样宽度（字节数），默认 2（16位 PCM）
    """
    # 读取原文件全部数据
    with open(input_file, 'rb') as f:
        data = f.read()

    if len(data) < 44:
        print(f"文件 {input_file} 太短，无法处理。")
        return

    # 跳过原有的 44 字节 header，保留 PCM 数据
    pcm_data = data[44:]

    # 使用 wave 模块写入新的 WAV 文件（会自动生成正确的 header）
    wf = wave.open(output_file, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(sampwidth)
    wf.setframerate(sample_rate)
    wf.writeframes(pcm_data)
    wf.close()

    print(f"处理完成: {input_file} -> {output_file}")

def process_directory(input_dir, output_dir, sample_rate=16000, channels=1, sampwidth=2):
    """
    遍历 input_dir 下所有 .wav 文件，调用 rebuild_wav_using_wave
    重新包装后保存到 output_dir 目录下。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 按文件名排序处理所有 WAV 文件
    wav_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith('.wav')])
    if not wav_files:
        print("未找到 WAV 文件。")
        return

    for wav_file in wav_files:
        input_path = os.path.join(input_dir, wav_file)
        output_path = os.path.join(output_dir, wav_file)
        rebuild_wav_using_wave(input_path, output_path, sample_rate, channels, sampwidth)

if __name__ == '__main__':
    # 修改下面的路径为你的实际路径
    input_directory = "/tmp/audio-d270954be32b448cb47a30630405e8fe"
    output_directory = "new_gen"
    process_directory(input_directory, output_directory)
