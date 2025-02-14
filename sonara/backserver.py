from http.server import HTTPServer, SimpleHTTPRequestHandler
from socketserver import TCPServer

import asyncio
import os
import subprocess
import ssl
import threading
import time
import uuid
import wave
import websockets


async def handle_connection(websocket):
    print("Client connected")
    # 为每个连接生成一个唯一的目录
    connection_hash = uuid.uuid4().hex
    directory = f"/tmp/audio-{connection_hash}"
    os.makedirs(directory, exist_ok=True)
    print(f"Audio files will be saved in: {directory}")

    counter = 1
    try:
        async for message in websocket:
            # 构造文件名，如 /tmp/audio-<hashcode>/00001.webm
            filename = os.path.join(directory, f"{counter:05d}.wav")
            with open(filename, "wb") as audio_file:
                audio_file.write(message)
            receive_time = time.time()
            print(f"Saved message {counter} ({len(message)} bytes) to {filename} at {receive_time}")
            counter += 1

        print("Connection closed. All audio messages have been saved.")
    except websockets.ConnectionClosed:
        print("Client disconnected")


async def start_websocket_server():
    async with websockets.serve(handle_connection, "0.0.0.0", 8765):
        print("WebSocket server started at ws://0.0.0.0:8765")
        await asyncio.Future()  # Run forever


def start_https_server():
    handler = SimpleHTTPRequestHandler
    server_address = ("0.0.0.0", 8080)  # HTTPS 监听 8080 端口
    httpd = HTTPServer(server_address, handler)

    print("HTTPS server started at https://0.0.0.0:8080")
    httpd.serve_forever()


# Run both servers
async def main():
    # Start the HTTP server in a separate thread
    http_thread = threading.Thread(target=start_https_server, daemon=True)
    http_thread.start()

    # Start the WebSocket server
    await start_websocket_server()


def main_entrypoint():
    import asyncio
    asyncio.run(main())


# Run the program
if __name__ == "__main__":
    print("Starting backserver")
    main_entrypoint()
