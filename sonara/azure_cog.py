import os
import json
import asyncio
import time
import uuid
import azure.cognitiveservices.speech as speechsdk
from sonara.groq_translator import GroqTranslator


class AzureCognitiveService:

    """
    wrap the azure continuous recognition logic, and send two types of messages through the websocket during the recognition process:
      - "recognizing": during the real-time recognition
      - "recognized": when a whole sentence is recognized
    """
    def __init__(self, websocket, loop):
        """
        :param websocket: the current websocket connection
        :param loop: asyncio event loop, for scheduling websocket send in the callback thread
        """
        self.websocket = websocket
        self.loop = loop
        
        # Debug variables
        self.debug_mode = os.getenv("DEBUG_TRANSLATION", "false").lower() == "true"
        self.processed_translations = []
        self.translation_times = {}
        
        # Create a translation queue for async processing
        self.translation_queue = asyncio.Queue()
        
        # Start the translation worker task
        print("Starting translation worker task...")
        self.translation_worker_task = self.loop.create_task(self.translation_worker())
        print(f"Translation worker task created: {self.translation_worker_task}")

        self.groq_translator = GroqTranslator(
            api_key=os.getenv("GROQ_API_KEY"),
            model=os.getenv("GROQ_MODEL")
        )

        # load azure config from environment variables
        speech_key = os.getenv("AZURE_SUBSCRIPTION_KEY")
        service_region = os.getenv("AZURE_REGION")
        if not speech_key or not service_region:
            raise ValueError("please ensure AZURE_SUBSCRIPTION_KEY and AZURE_REGION are set in environment variables")
        
        speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
        speech_config.speech_recognition_language = "en-US"  # or other language settings
        speech_config.enable_automatic_punctuation = True

        # Enable speaker separation in intermediate results
        speech_config.set_property(
            property_id=speechsdk.PropertyId.SpeechServiceResponse_DiarizeIntermediateResults,
            value='true'
        )

        # create a push audio stream and create an AudioConfig based on it
        self.push_stream = speechsdk.audio.PushAudioInputStream()
        audio_config = speechsdk.audio.AudioConfig(stream=self.push_stream)
        
        # Create a conversation transcriber (instead of speech recognizer)
        self.conversation_transcriber = speechsdk.transcription.ConversationTranscriber(
            speech_config=speech_config,
            audio_config=audio_config
        )
        
        # Register conversation transcription events
        self.conversation_transcriber.transcribing.connect(self.handle_transcribing)
        self.conversation_transcriber.transcribed.connect(self.handle_transcribed)  # ensure this method exists
        self.conversation_transcriber.canceled.connect(lambda evt: print(f"Transcription canceled: {evt}"))
        self.conversation_transcriber.session_started.connect(lambda evt: print("Session started"))
        self.conversation_transcriber.session_stopped.connect(lambda evt: print("Session ended"))
        
        # Test the translation queue when in debug mode
        if self.debug_mode:
            print("*** TRANSLATION QUEUE DEBUG MODE ENABLED ***")
            asyncio.run_coroutine_threadsafe(self.run_translation_test(), self.loop)
        
        # Start continuous transcription
        self.conversation_transcriber.start_transcribing_async()
        print("Azure conversation transcriber started")
    
    def handle_transcribing(self, evt):
        """Real-time transcription callback"""
        if evt.result.text:
            # Get speaker ID and speaker information
            speaker_id = "unknown"
            if hasattr(evt.result, "speaker_id"):
                speaker_id = evt.result.speaker_id
            elif hasattr(evt.result, "speaker") and evt.result.speaker:
                speaker_id = evt.result.speaker
            
            message = json.dumps({
                "type": "recognizing", 
                "result": evt.result.text,
                "speaker": speaker_id
            })
            asyncio.run_coroutine_threadsafe(self.websocket.send(message), self.loop)
            print(f"Sending real-time transcription result: {evt.result.text}, Speaker: {speaker_id}")
        
    def handle_transcribed(self, evt):
        """Final transcription callback"""
        if not evt.result.text:
            return
        
        # Get speaker ID
        speaker_id = "unknown"
        if hasattr(evt.result, "speaker_id"):
            speaker_id = evt.result.speaker_id
        elif hasattr(evt.result, "speaker") and evt.result.speaker:
            speaker_id = evt.result.speaker
        
        message = json.dumps({
            "type": "recognized", 
            "result": evt.result.text,
            "speaker": speaker_id
        })
        asyncio.run_coroutine_threadsafe(self.websocket.send(message), self.loop)
        print(f"Sending final transcription result: {evt.result.text}, Speaker: {speaker_id}")
        
        # Generate a short unique ID for this translation task
        task_id = str(uuid.uuid4())[:8]
        
        # Use asyncio.create_task on the event loop to start the enqueue_translation task
        # This ensures the task runs properly in the event loop context
        print(f"Creating enqueue_translation task with ID: {task_id}")
        future = asyncio.run_coroutine_threadsafe(
            self.enqueue_translation(evt.result.text, speaker_id, task_id), 
            self.loop
        )
        
        # Use a callback to log completion or errors rather than blocking
        def done_callback(fut):
            try:
                fut.result()
                print(f"Successfully enqueued translation task {task_id}")
            except Exception as e:
                print(f"Failed to enqueue translation task {task_id}: {e}")
                
        future.add_done_callback(done_callback)
    
    async def enqueue_translation(self, text: str, speaker_id="unknown", task_id=None):
        """
        Add a translation task to the queue
        """
        if not task_id:
            task_id = str(uuid.uuid4())[:8]
            
        print(f"[{task_id}] Enqueuing translation: '{text}' for speaker {speaker_id}")
        
        # Store the timestamp when this task was added
        self.translation_times[task_id] = {
            "text": text,
            "speaker_id": speaker_id,
            "enqueued_at": time.time()
        }
        
        # Put the task in the queue
        await self.translation_queue.put((text, speaker_id, task_id))
        
        queue_size = self.translation_queue.qsize()
        print(f"[{task_id}] Added to translation queue. Current queue size: {queue_size}")
    
    async def translation_worker(self):
        """
        Worker that processes translation tasks from the queue
        """
        print("Translation worker started!")
        try:
            while True:
                print(f"Translation worker waiting for next task... Queue size: {self.translation_queue.qsize()}")
                # Wait for a translation task
                text, speaker_id, task_id = await self.translation_queue.get()
                
                start_time = time.time()
                if task_id in self.translation_times:
                    queue_wait_time = start_time - self.translation_times[task_id]["enqueued_at"]
                    print(f"[{task_id}] Starting translation after {queue_wait_time:.2f}s in queue")
                else:
                    print(f"[{task_id}] Starting translation (no timing data available)")
                
                try:
                    # Perform the translation
                    print(f"[{task_id}] Calling groq_translator.translate_with_retries for: '{text}'")
                    translation = await self.loop.run_in_executor(
                        None, 
                        self.groq_translator.translate_with_retries, 
                        text
                    )
                    
                    end_time = time.time()
                    duration = end_time - start_time
                    
                    print(f"[{task_id}] Translation completed in {duration:.2f}s: '{translation}'")
                    
                    # Check if websocket is connected using our dedicated method
                    websocket_connected = await self.is_websocket_connected()
                    print(f"[{task_id}] Websocket connected: {websocket_connected}")

                    if translation and websocket_connected:
                        translated_message = json.dumps({
                            "type": "translated", 
                            "result": translation,
                            "speaker": speaker_id
                        })
                        print(f"[{task_id}] Sending translated message: {translated_message}")
                        try:
                            await self.websocket.send(translated_message)
                            print(f"[{task_id}] Sent translation result (took {duration:.2f}s): {translation}, Speaker: {speaker_id}")
                            
                            # Record for debugging
                            if task_id in self.translation_times:
                                self.translation_times[task_id]["completed_at"] = end_time
                                self.translation_times[task_id]["duration"] = duration
                                self.translation_times[task_id]["translation"] = translation
                                self.processed_translations.append(self.translation_times[task_id])
                        except Exception as e:
                            print(f"[{task_id}] Failed to send translation via websocket: {e}")
                    else:
                        print(f"[{task_id}] Translation completed but unable to send to frontend")
                        print(f"[{task_id}] translation: '{translation}', websocket connected: {websocket_connected}")
                except Exception as e:
                    print(f"[{task_id}] Translation failed for text '{text}': {e}")
                    if task_id in self.translation_times:
                        self.translation_times[task_id]["error"] = str(e)
                
                # Mark the task as done
                self.translation_queue.task_done()
                print(f"[{task_id}] Task completed. Remaining queue size: {self.translation_queue.qsize()}")
        except asyncio.CancelledError:
            print("Translation worker was cancelled")
        except Exception as e:
            print(f"Unexpected error in translation worker: {e}")
            import traceback
            traceback.print_exc()
    
    async def call_translation(self, text: str, speaker_id="unknown"):
        """
        Call async translation function and send translation results to frontend via websocket
        """
        # This method is kept for backward compatibility
        # Now it just calls enqueue_translation
        await self.enqueue_translation(text, speaker_id)
    
    async def run_translation_test(self):
        """
        Test method to verify the translation queue is working correctly
        """
        print("\n*** STARTING TRANSLATION QUEUE TEST ***")
        test_sentences = [
            "This is the first test sentence.",
            "Here's a second sentence to translate.",
            "And a third one to verify order is maintained.",
            "Finally, the fourth sentence should come last."
        ]
        
        print(f"Enqueueing {len(test_sentences)} test translations...")
        for i, sentence in enumerate(test_sentences):
            test_id = f"TEST-{i+1}"
            await self.enqueue_translation(sentence, f"test-speaker-{i+1}", test_id)
            # Small delay to ensure obvious ordering
            await asyncio.sleep(0.1)
            
        print("All test translations enqueued. Waiting for processing...")
        # Wait for all translations to complete
        await self.translation_queue.join()
        print("\n*** TRANSLATION QUEUE TEST COMPLETED ***")
        print(f"Processed {len(self.processed_translations)} translations")
        for i, result in enumerate(self.processed_translations):
            print(f"Result {i+1}:")
            print(f"  Original: {result['text']}")
            print(f"  Translation: {result.get('translation', 'N/A')}")
            if 'enqueued_at' in result and 'completed_at' in result:
                total_time = result['completed_at'] - result['enqueued_at']
                print(f"  Total time: {total_time:.2f}s (Translation: {result.get('duration', 'N/A'):.2f}s)")
            if 'error' in result:
                print(f"  Error: {result['error']}")
            print("")

    def write(self, data: bytes):
        """
        write the received audio data to the push stream
        """
        self.push_stream.write(data)
    
    def close(self):
        """
        close the push stream and stop the recognizer
        """
        # Cancel the translation worker task
        if hasattr(self, 'translation_worker_task') and self.translation_worker_task:
            print("Cancelling translation worker task...")
            self.translation_worker_task.cancel()
            print("Translation worker task cancelled")
            
        # Print translation statistics if we're in debug mode
        if self.debug_mode and self.processed_translations:
            print("\n*** TRANSLATION STATISTICS ***")
            print(f"Total translations processed: {len(self.processed_translations)}")
            
        self.push_stream.close()
        self.conversation_transcriber.stop_transcribing_async()
        print("Azure speech recognizer stopped")

    async def is_websocket_connected(self):
        """
        Check if the websocket is connected by examining its properties
        """
        if not hasattr(self, 'websocket'):
            return False
            
        try:
            # First check our custom property if it exists
            if hasattr(self.websocket, 'custom_is_open'):
                return self.websocket.custom_is_open
                
            # Different websocket implementations have different ways to check connection status
            # Try various attributes that might indicate connection status
            if hasattr(self.websocket, 'open'):
                return self.websocket.open
            elif hasattr(self.websocket, 'closed'):
                return not self.websocket.closed
            elif hasattr(self.websocket, 'state') and hasattr(self.websocket.state, 'value'):
                # Some implementations use a state enum
                return self.websocket.state.value == 1  # 1 often means OPEN
            else:
                # As a last resort, try a simple ping to see if the connection is alive
                try:
                    # For internal use only, try a simple attribute access to see if the object is valid
                    # This won't actually check connection state but might catch some dead references
                    str(self.websocket)
                    return True
                except Exception:
                    return False
        except Exception as e:
            print(f"Error checking websocket connection: {e}")
            return False
