import asyncio
import websockets
import wave
from http.server import SimpleHTTPRequestHandler
from socketserver import TCPServer
import threading
import subprocess

# WebSocket server handler
async def handle_connection(websocket, path):
    print("Client connected")
    try:
        # Open a WAV file for writing
        with wave.open("output.wav", "wb") as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit audio
            wav_file.setframerate(44100)  # 44.1 kHz sample rate

            async for message in websocket:
                print(f"Received audio data: {len(message)} bytes")
                wav_file.writeframes(message)  # Write audio data to the file

        # Play the audio file using the playsound.py script
        subprocess.run(["python", "playsound.py"])
    except websockets.ConnectionClosed:
        print("Client disconnected")

# Start the WebSocket server
async def start_websocket_server():
    async with websockets.serve(handle_connection, "localhost", 8765):
        print("WebSocket server started at ws://localhost:8765")
        await asyncio.Future()  # Run forever

# Start the HTTP server
def start_http_server():
    handler = SimpleHTTPRequestHandler
    httpd = TCPServer(("localhost", 8080), handler)  # Changed port to 8080
    print("HTTP server started at http://localhost:8080")
    httpd.serve_forever()

# Run both servers
async def main():
    # Start the HTTP server in a separate thread
    http_thread = threading.Thread(target=start_http_server, daemon=True)
    http_thread.start()

    # Start the WebSocket server
    await start_websocket_server()

# Run the program
if __name__ == "__main__":
    asyncio.run(main())