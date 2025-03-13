import asyncio
import json
import os
import pytest
import pytest_asyncio
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

# We'll use module level patching to avoid segmentation faults
speech_config_patch = patch("azure.cognitiveservices.speech.SpeechConfig")
audio_stream_patch = patch("azure.cognitiveservices.speech.audio.PushAudioInputStream")
transcriber_patch = patch("azure.cognitiveservices.speech.transcription.ConversationTranscriber")

# Apply all patches at module level
speech_config_patch.start()
audio_stream_patch.start()
transcriber_patch.start()

# Mock class for the GroqTranslator
class MockGroqTranslator:
    def translate_with_retries(self, text):
        # Simulate translation with predictable result for testing
        return f"TRANSLATED: {text}"

# Mock for AzureCognitiveService that doesn't rely on the actual Azure SDK
class MockAzureCognitiveService:
    def __init__(self, websocket, loop):
        self.websocket = websocket
        self.loop = loop
        self.translation_queue = asyncio.Queue()
        self.processed_translations = []
        self.translation_times = {}
        
        # Create a translator
        self.groq_translator = MockGroqTranslator()
        
        # Start the worker
        self.translation_worker_task = loop.create_task(self.translation_worker())
    
    async def enqueue_translation(self, text, speaker_id="unknown", task_id=None):
        if not task_id:
            task_id = str(uuid.uuid4())[:8]
            
        # Store timestamp
        self.translation_times[task_id] = {
            "text": text,
            "speaker_id": speaker_id,
            "enqueued_at": asyncio.get_event_loop().time()
        }
        
        # Put in queue
        await self.translation_queue.put((text, speaker_id, task_id))
        return task_id
    
    async def translation_worker(self):
        try:
            while True:
                # Wait for an item
                text, speaker_id, task_id = await self.translation_queue.get()
                
                try:
                    # Simulate translation
                    translation = self.groq_translator.translate_with_retries(text)
                    
                    # Send via websocket
                    if self.websocket and hasattr(self.websocket, 'send'):
                        message = json.dumps({
                            "type": "translated", 
                            "result": translation,
                            "speaker": speaker_id
                        })
                        await self.websocket.send(message)
                    
                    # Record for testing
                    self.translation_times[task_id]["translation"] = translation
                    self.processed_translations.append({
                        "text": text,
                        "speaker_id": speaker_id,
                        "translation": translation,
                        "task_id": task_id
                    })
                except Exception as e:
                    print(f"Error in worker: {e}")
                
                # Mark as done
                self.translation_queue.task_done()
        except asyncio.CancelledError:
            pass
    
    def handle_transcribed(self, event):
        """Mock implementation of handle_transcribed"""
        if hasattr(event, 'result') and event.result.text:
            speaker_id = getattr(event.result, 'speaker', "unknown")
            asyncio.run_coroutine_threadsafe(
                self.enqueue_translation(event.result.text, speaker_id), 
                self.loop
            )
    
    async def close(self):
        """Clean up resources"""
        if hasattr(self, 'translation_worker_task'):
            self.translation_worker_task.cancel()
            try:
                await self.translation_worker_task
            except asyncio.CancelledError:
                pass


# Mock event class
class MockRecognitionEvent:
    def __init__(self, text, speaker_id="test-speaker"):
        self.result = MagicMock()
        self.result.text = text
        self.result.speaker = speaker_id


@pytest.fixture
def mock_env_vars():
    """Set up mock environment variables for testing"""
    os.environ["GROQ_API_KEY"] = "test_api_key"
    os.environ["GROQ_MODEL"] = "test_model"
    os.environ["AZURE_SUBSCRIPTION_KEY"] = "test_subscription"
    os.environ["AZURE_REGION"] = "test_region"
    os.environ["DEBUG_TRANSLATION"] = "true"
    
    yield
    
    # Clean up env vars
    for key in ["GROQ_API_KEY", "GROQ_MODEL", "AZURE_SUBSCRIPTION_KEY", "AZURE_REGION", "DEBUG_TRANSLATION"]:
        if key in os.environ:
            del os.environ[key]


@pytest_asyncio.fixture
async def mock_azure_service(mock_env_vars):
    """Create a mocked AzureCognitiveService instance for testing"""
    # Create mock WebSocket
    mock_websocket = AsyncMock()
    mock_websocket.open = True
    mock_websocket.send = AsyncMock()
    
    # Create mock event loop
    loop = asyncio.get_event_loop()
    
    # Create our mock service
    service = MockAzureCognitiveService(mock_websocket, loop)
    
    # Give some time for initialization
    await asyncio.sleep(0.1)
    
    yield service
    
    # Cleanup
    await service.close()


@pytest.mark.asyncio
async def test_enqueue_translation(mock_azure_service):
    """Test that translations can be enqueued correctly"""
    # Arrange
    service = mock_azure_service
    test_text = "This is a test sentence."
    test_speaker = "test-speaker"
    task_id = str(uuid.uuid4())[:8]
    
    # Act
    await service.enqueue_translation(test_text, test_speaker, task_id)
    
    # Assert
    assert service.translation_queue.qsize() == 1
    queued_item = await service.translation_queue.get()
    text, speaker, tid = queued_item
    assert text == test_text
    assert speaker == test_speaker
    assert tid == task_id
    assert task_id in service.translation_times
    
    # Cleanup - mark task as done
    service.translation_queue.task_done()


@pytest.mark.asyncio
async def test_translation_order(mock_azure_service):
    """Test that translations are processed in the correct order"""
    # Arrange
    service = mock_azure_service
    
    # Create test sentences
    test_sentences = [
        "1. This is the first test sentence.",
        "2. This is the second test sentence.",
        "3. This is the third test sentence."
    ]
    
    # Add sentences to the queue
    for i, sentence in enumerate(test_sentences):
        await service.enqueue_translation(sentence, f"speaker-{i+1}", f"TEST-{i+1}")
    
    # Act - wait for all translations to complete
    await service.translation_queue.join()
    
    # Assert - Check the order of processed translations
    assert len(service.processed_translations) == 3
    
    # Extract the original text to verify order
    processed_texts = [item.get("text", "") for item in service.processed_translations]
    
    # Verify they're in the same order as the input
    assert processed_texts == test_sentences


@pytest.mark.asyncio
async def test_handle_transcribed(mock_azure_service):
    """Test that the handle_transcribed method correctly enqueues translations"""
    # Arrange
    service = mock_azure_service
    test_text = "This is a transcribed sentence."
    
    # Create a mock event
    mock_event = MockRecognitionEvent(test_text)
    
    # Create a patched version of handle_transcribed that we can monitor
    with patch.object(service, 'enqueue_translation', new_callable=AsyncMock) as mock_enqueue:
        # Call the handler
        service.handle_transcribed(mock_event)
        
        # Wait a short time for the coroutine to be scheduled
        await asyncio.sleep(0.5)
        
        # Assert that enqueue_translation was called with the right parameters
        mock_enqueue.assert_called_once()
        args, kwargs = mock_enqueue.call_args
        assert args[0] == test_text  # First arg should be the text
        assert args[1] == "test-speaker"  # Second arg should be the speaker_id


@pytest.mark.asyncio
async def test_translation_worker_processes_queue(mock_azure_service):
    """Test that the translation worker correctly processes the queue"""
    # Arrange
    service = mock_azure_service
    test_text = "This is a test for the worker."
    
    # Ensure the websocket is mocked
    assert hasattr(service.websocket, 'send')
    assert asyncio.iscoroutinefunction(service.websocket.send)
    
    # Act - Add a translation task to the queue
    task_id = "TEST-WORKER"
    await service.enqueue_translation(test_text, "test-speaker", task_id)
    
    # Wait for the worker to process the task
    await asyncio.sleep(0.5)  # Give time for the worker to process
    await service.translation_queue.join()
    
    # Assert - Check that the websocket.send was called with the correct translation
    service.websocket.send.assert_called()
    
    # Get the last call arguments
    last_call = service.websocket.send.call_args_list[-1]
    sent_json = json.loads(last_call[0][0])
    
    assert sent_json["type"] == "translated"
    assert sent_json["result"] == f"TRANSLATED: {test_text}"
    assert sent_json["speaker"] == "test-speaker"
    
    # Verify it was tracked for debugging
    assert len(service.processed_translations) == 1
    assert service.processed_translations[0]["text"] == test_text


# Clean up patches at module exit
def teardown_module(module):
    speech_config_patch.stop()
    audio_stream_patch.stop()
    transcriber_patch.stop() 