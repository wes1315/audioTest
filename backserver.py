import asyncio
import websockets
import wave
from http.server import HTTPServer, SimpleHTTPRequestHandler
from socketserver import TCPServer
import threading
import subprocess
import ssl


# WebSocket server handler
async def handle_connection(websocket):
    print("Client connected")
    try:
        # 打开 WebM 文件用于写入
        with open("/tmp/output.webm", "wb") as audio_file:
            async for message in websocket:
                print(f"Received audio data: {len(message)} bytes")
                audio_file.write(message)  # 直接写入 WebM 格式的音频数据

        print("Recording saved to /tmp/output.webm")
        # 播放录制的音频（使用 ffplay）
        # subprocess.run(["ffplay", "-nodisp", "-autoexit", "/tmp/output.webm"])
    
    except websockets.ConnectionClosed:
        print("Client disconnected")


async def start_websocket_server():
    ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ssl_context.load_cert_chain(certfile="ssl/server.crt", keyfile="ssl/server.key")

    async with websockets.serve(handle_connection, "0.0.0.0", 8765):
        print("WebSocket server started at ws://0.0.0.0:8765")
        await asyncio.Future()  # Run forever


def start_https_server():
    handler = SimpleHTTPRequestHandler
    server_address = ("0.0.0.0", 8443)  # HTTPS 监听 8443 端口
    httpd = HTTPServer(server_address, handler)

    # 创建 SSL 上下文
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(certfile="ssl/server.crt", keyfile="ssl/server.key")

    # 使用新的方式封装 SSL
    httpd.socket = context.wrap_socket(httpd.socket, server_side=True)

    print("HTTPS server started at https://0.0.0.0:8443")
    httpd.serve_forever()


# Run both servers
async def main():
    # Start the HTTP server in a separate thread
    http_thread = threading.Thread(target=start_https_server, daemon=True)
    http_thread.start()

    # Start the WebSocket server
    await start_websocket_server()


# Run the program
if __name__ == "__main__":
    asyncio.run(main())
