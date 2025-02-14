#!/usr/bin/env python3
import os
import av
import numpy as np
import wave

def binary_concat_webm_files(input_directory, output_file):
    """
    将 input_directory 中所有 .webm 文件按文件名排序后，
    以二进制方式顺序拼接成一个大文件 output_file。
    """
    files = sorted([f for f in os.listdir(input_directory) if f.endswith('.webm')])
    print(f"找到 {len(files)} 个 .webm 文件：", files)
    with open(output_file, 'wb') as out_f:
        for f in files:
            path = os.path.join(input_directory, f)
            print(f"读取文件: {path}")
            with open(path, 'rb') as in_f:
                data = in_f.read()
                out_f.write(data)
    print(f"二进制拼接完成，共拼接 {len(files)} 个文件到 {output_file}")

def segment_webm_to_wav(combined_file, output_dir, segment_duration=1.0):
    """
    利用 PyAV 将 combined_file（拼接后的 WebM 文件）解码，
    并按 segment_duration（秒）分割成多个 WAV 文件，
    输出文件命名为 segment_00001.wav, segment_00002.wav, ...
    
    增加了调试输出，打印每个 frame 的关键信息及累计样本数，
    方便检查问题所在。
    """
    debug = True  # 打开调试输出
    try:
        container = av.open(combined_file)
    except Exception as e:
        print("打开拼接后的文件失败:", e)
        return

    if not container.streams.audio:
        print("未找到音频流！")
        return

    audio_stream = container.streams.audio[0]
    sample_rate = audio_stream.rate
    print("检测到采样率:", sample_rate)

    # 用于累计解码后得到的音频数据（假设为浮点32类型，单声道数据）
    samples_accum = np.array([], dtype=np.float32)
    segment_idx = 1

    for frame in container.decode(audio=0):
        if debug:
            print("-" * 60)
            print(f"处理新 frame: pts={frame.pts}, dts={frame.dts}, time={frame.time}, samples={frame.samples}")
            # 打印音频帧的格式和通道信息
            print("frame.format:", frame.format.name)
            print("frame.layout:", frame.layout.name if frame.layout else "无")
        
        try:
            arr = frame.to_ndarray()
        except Exception as e:
            print("frame.to_ndarray() 失败:", e)
            continue

        if debug:
            print("转换后的数组形状:", arr.shape, "dtype:", arr.dtype)
        
        # 如果有多声道，取平均生成单声道数据
        if arr.ndim > 1 and arr.shape[0] > 1:
            arr = arr.mean(axis=0)
            if debug:
                print("多声道数据，取平均后形状:", arr.shape)
        else:
            arr = arr.flatten()
            if debug:
                print("单声道或已扁平化数据，形状:", arr.shape)

        # 确保数据是 float32 类型，如果不是，尝试转换
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32)
            if debug:
                print("转换为 float32 类型")
        
        samples_accum = np.concatenate([samples_accum, arr])
        if debug:
            print(f"累计样本数: {len(samples_accum)} (目标每段 {int(sample_rate * segment_duration)} 样本)")
        
        # 当累计的样本数达到或超过 1 秒所需样本数时，输出一段 WAV 文件
        while len(samples_accum) >= int(sample_rate * segment_duration):
            num_samples = int(sample_rate * segment_duration)
            segment_samples = samples_accum[:num_samples]
            samples_accum = samples_accum[num_samples:]
            if debug:
                print(f"生成 segment {segment_idx:05d}，样本数: {len(segment_samples)}")
                print("segment_samples 的统计: min={:.3f}, max={:.3f}, mean={:.3f}".format(
                    segment_samples.min(), segment_samples.max(), segment_samples.mean()))
            
            # 这里假设解码后的浮点数在 [-1,1]，转换为 16 位 PCM
            int_samples = np.int16(np.clip(segment_samples, -1, 1) * 32767)
            out_filename = os.path.join(output_dir, f"segment_{segment_idx:05d}.wav")
            try:
                with wave.open(out_filename, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)       # 16 位 PCM 占 2 字节
                    wf.setframerate(sample_rate)
                    wf.writeframes(int_samples.tobytes())
                print(f"已写入 {out_filename}")
            except Exception as e:
                print(f"写入 {out_filename} 时发生错误:", e)
            segment_idx += 1

if __name__ == "__main__":
    # 1. 二进制拼接（根据你的描述，拼接是正确的）
    input_directory = "/tmp/audio-4e6f4960cdf6455f8b0749b9362b4ef6"  # 存放 .webm 文件的目录
    combined_file = "combined.webm"
    binary_concat_webm_files(input_directory, combined_file)

    # 2. 分段输出 WAV 文件
    output_dir = "segments"
    os.makedirs(output_dir, exist_ok=True)
    segment_webm_to_wav(combined_file, output_dir, segment_duration=1.0)
