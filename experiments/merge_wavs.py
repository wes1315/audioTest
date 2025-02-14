import os
import struct

def merge_wav_files(input_dir, output_file):
    # 获取目录下所有以 .wav 结尾的文件，按文件名排序
    wav_files = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.lower().endswith('.wav')])
    if not wav_files:
        print("目录中没有找到 WAV 文件")
        return

    all_pcm_data = b""

    # 读取第一个文件：保留 header 和 PCM 数据
    with open(wav_files[0], 'rb') as f:
        header = f.read(44)  # WAV 文件头部 44 字节
        pcm_data = f.read()  # PCM 数据
        all_pcm_data += pcm_data

    # 后续文件跳过前 44 字节头部，只保留 PCM 数据
    for wav_file in wav_files[1:]:
        with open(wav_file, 'rb') as f:
            f.read(44)  # 跳过头部
            pcm_data = f.read()
            all_pcm_data += pcm_data

    # 重新计算文件大小信息
    # 整个文件大小（不含前 8 字节） = 36 + PCM 数据大小
    new_chunk_size = 36 + len(all_pcm_data)
    # 数据块大小即 PCM 数据大小
    data_chunk_size = len(all_pcm_data)

    # 修改第一个文件的 header 中的文件大小和数据块大小
    new_header = bytearray(header)
    struct.pack_into('<I', new_header, 4, new_chunk_size)   # 文件大小字段（不含前 8 字节）
    struct.pack_into('<I', new_header, 40, data_chunk_size)   # 数据块大小

    # 写入合并后的文件
    with open(output_file, 'wb') as out_f:
        out_f.write(new_header)
        out_f.write(all_pcm_data)

    print(f"合并成功，输出文件：{output_file}")

if __name__ == '__main__':
    # 修改这里为你的存放 WAV 文件的目录和合并后的输出文件名
    input_directory = "/tmp/audio-d270954be32b448cb47a30630405e8fe"
    output_filename = "merged.wav"
    merge_wav_files(input_directory, output_filename)
