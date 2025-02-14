#!/usr/bin/env python3
import os
import gi

gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

# 初始化 GStreamer（如果已初始化可以忽略）
Gst.init(None)


def convert_block_to_wav(header_data: bytes, chunk_data: bytes) -> bytes:
    """
    利用 GStreamer 的 appsrc/appsink 将一段 WebM 数据流转换为 WAV 数据。
    header_data 为前置的 header（例如提取自 00001.webm），
    chunk_data 为当前块数据（可能仅包含部分数据，不含完整头部）。
    对于完整的 WebM 文件，可以传入 header_data = b""。
    返回转换后的 WAV 文件二进制数据。
    """
    pipeline_str = (
        'appsrc name=mysrc format=bytes caps="audio/x-matroska, stream-format=webm, alignment=(string)au, codecs=(string)A_OPUS" ! '
        'matroskademux ! queue ! avdec_opus ! audioconvert ! audioresample ! wavenc ! appsink name=mysink'
    )
    pipeline = Gst.parse_launch(pipeline_str)
    appsrc = pipeline.get_by_name("mysrc")
    appsink = pipeline.get_by_name("mysink")

    output_data = bytearray()

    def on_new_sample(sink, data):
        sample = sink.emit("pull-sample")
        if sample:
            buf = sample.get_buffer()
            result, map_info = buf.map(Gst.MapFlags.READ)
            if result:
                output_data.extend(map_info.data)
                buf.unmap(map_info)
        return Gst.FlowReturn.OK

    appsink.connect("new-sample", on_new_sample, None)

    pipeline.set_state(Gst.State.PLAYING)

    # 如果 header_data 非空，先推入 header_data
    if header_data:
        buf = Gst.Buffer.new_allocate(None, len(header_data), None)
        buf.fill(0, header_data)
        ret = appsrc.emit("push-buffer", buf)
        if ret != Gst.FlowReturn.OK:
            print("Error pushing header buffer:", ret)
    # 推入当前块数据
    buf = Gst.Buffer.new_allocate(None, len(chunk_data), None)
    buf.fill(0, chunk_data)
    ret = appsrc.emit("push-buffer", buf)
    if ret != Gst.FlowReturn.OK:
        print("Error pushing chunk buffer:", ret)
    appsrc.emit("end-of-stream")

    bus = pipeline.get_bus()
    while True:
        msg = bus.timed_pop_filtered(500 * Gst.MSECOND,
                                     Gst.MessageType.ERROR | Gst.MessageType.EOS)
        if msg:
            if msg.type == Gst.MessageType.ERROR:
                err, debug = msg.parse_error()
                print("Pipeline error:", err, debug)
                break
            elif msg.type == Gst.MessageType.EOS:
                break

    pipeline.set_state(Gst.State.NULL)
    return bytes(output_data)


def main():
    # 假设 WebM 文件存放在此目录
    directory = "/tmp/audio-4e6f4960cdf6455f8b0749b9362b4ef6"
    files = sorted([f for f in os.listdir(directory) if f.endswith(".webm")])
    if not files:
        print("目录中未找到任何 .webm 文件")
        return

    # 读取 header 文件 00001.webm，假设其中包含完整 header 信息和部分音频数据
    header_filename = "00001.webm"
    header_path = os.path.join(directory, header_filename)
    try:
        with open(header_path, "rb") as f:
            header_full_data = f.read()
    except Exception as e:
        print(f"读取 {header_filename} 出错: {e}")
        return

    # 从 header_full_data 中提取 header 元数据
    # 这里简单采用查找第一个 Cluster 标记 (0x1F43B675)
    marker = b'\x1F\x43\xB6\x75'
    index = header_full_data.find(marker)
    if index != -1:
        header_metadata = header_full_data[:index]
        print(f"提取到 header 元数据, 长度: {len(header_metadata)} 字节")
    else:
        header_metadata = header_full_data
        print("未找到 Cluster 标记, 使用完整 header 数据")

    # 针对每个文件分别转换为 WAV 文件
    for f_name in files:
        in_path = os.path.join(directory, f_name)
        out_path = os.path.join(directory, f_name.replace(".webm", ".wav"))
        try:
            with open(in_path, "rb") as f:
                file_data = f.read()
        except Exception as e:
            print(f"读取 {f_name} 出错: {e}")
            continue

        # 如果当前文件是 header 文件，则不再推入 header（假设其已完整），否则推入 header_metadata + 块数据
        if f_name == header_filename:
            wav_data = convert_block_to_wav(b"", file_data)
        else:
            wav_data = convert_block_to_wav(header_metadata, file_data)

        try:
            with open(out_path, "wb") as f:
                f.write(wav_data)
            print(f"转换完成: {f_name} --> {os.path.basename(out_path)}")
        except Exception as e:
            print(f"写入 {out_path} 出错: {e}")

if __name__ == "__main__":
    main()
