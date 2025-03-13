from http.server import HTTPServer, SimpleHTTPRequestHandler
import logging
from socketserver import TCPServer

import asyncio
import os
import subprocess
import ssl
import threading
import time
import uuid
import wave
from dotenv import load_dotenv
import websockets
from sonara.azure_cog import AzureCognitiveService


class WebSocketWrapper:
    """A wrapper around a websocket connection to ensure proper state tracking"""
    def __init__(self, websocket):
        self.websocket = websocket
        self.is_open = True
    
    async def send(self, message):
        if self.is_open:
            await self.websocket.send(message)
        else:
            print("Warning: Attempted to send message on closed websocket")
    
    def close(self):
        self.is_open = False
    
    @property
    def open(self):
        return self.is_open


async def handle_connection(websocket):
    print("client connected")
    # generate a unique directory for each connection, for saving wav files
    connection_hash = uuid.uuid4().hex
    directory = f"/tmp/audio-{connection_hash}"
    os.makedirs(directory, exist_ok=True)
    print(f"audio files will be saved in: {directory}")

    # Wrap the websocket to ensure proper state tracking
    wrapped_websocket = WebSocketWrapper(websocket)
    
    # get the current event loop, and initialize the azure recognition service
    loop = asyncio.get_running_loop()
    azure_service = AzureCognitiveService(wrapped_websocket, loop)
    
    counter = 1
    try:
        async for message in websocket:
            # save each received audio chunk as a .wav file
            filename = os.path.join(directory, f"{counter:05d}.wav")
            with open(filename, "wb") as audio_file:
                audio_file.write(message)
            receive_time = time.time()
            print(f"saved message {counter} (size: {len(message)} bytes) to {filename}, timestamp: {receive_time}")

            # push the data to the azure push stream
            azure_service.write(message)            
            counter += 1

        print("connection closed, all audio messages saved")
    except websockets.ConnectionClosed:
        print("client disconnected")
    except Exception as e:
        print(f"Error in websocket connection handler: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Mark the websocket as closed
        wrapped_websocket.close()
            
        # close the azure push stream and recognizer
        azure_service.close()


async def start_websocket_server():
    async with websockets.serve(handle_connection, "0.0.0.0", 8765):
        print("WebSocket server started at ws://0.0.0.0:8765")
        await asyncio.Future()  # Run forever


def start_https_server():
    handler = SimpleHTTPRequestHandler
    server_address = ("0.0.0.0", 8080)
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
    load_dotenv()
    # logging.basicConfig(level=logging.DEBUG)

    import asyncio
    asyncio.run(main())


# Run the program
if __name__ == "__main__":
    print("Starting backserver")
    main_entrypoint()
