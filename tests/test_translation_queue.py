import asyncio
import os
import pytest
import pytest_asyncio
import time
from unittest.mock import AsyncMock, MagicMock

# A simplified version of the translation queue system for testing
class TranslationQueueService:
    def __init__(self, websocket):
        self.websocket = websocket
        self.translation_queue = asyncio.Queue()
        self.processed_items = []
        self.processing_times = {}
        
        # Start the worker task
        self.worker_task = asyncio.create_task(self.translation_worker())
    
    async def enqueue_item(self, text, speaker_id, task_id):
        """Add an item to the queue"""
        print(f"[{task_id}] Enqueuing: '{text}' for speaker {speaker_id}")
        
        # Store the timestamp
        self.processing_times[task_id] = {
            "text": text,
            "speaker_id": speaker_id,
            "enqueued_at": time.time()
        }
        
        # Put in queue
        await self.translation_queue.put((text, speaker_id, task_id))
        return task_id
    
    async def translation_worker(self):
        """Process items from the queue"""
        try:
            while True:
                # Wait for an item
                text, speaker_id, task_id = await self.translation_queue.get()
                
                start_time = time.time()
                print(f"[{task_id}] Processing item: {text}")
                
                try:
                    # Simulate translation - longer for first item
                    if "1" in text:
                        await asyncio.sleep(0.2)
                    else:
                        await asyncio.sleep(0.1)
                    
                    # Create simulated translation
                    translation = f"TRANSLATED: {text}"
                    
                    # Send result via websocket
                    if self.websocket and hasattr(self.websocket, 'send'):
                        await self.websocket.send(translation)
                    
                    # Store result for verification
                    end_time = time.time()
                    if task_id in self.processing_times:
                        self.processing_times[task_id]["completed_at"] = end_time
                        self.processing_times[task_id]["duration"] = end_time - start_time
                        self.processing_times[task_id]["translation"] = translation
                        self.processed_items.append(self.processing_times[task_id])
                        
                except Exception as e:
                    print(f"Error processing task {task_id}: {e}")
                
                # Mark task as done
                self.translation_queue.task_done()
        except asyncio.CancelledError:
            print("Worker task cancelled")
    
    async def close(self):
        """Clean up resources"""
        if self.worker_task:
            self.worker_task.cancel()
            try:
                await self.worker_task
            except asyncio.CancelledError:
                pass

@pytest_asyncio.fixture
async def queue_service():
    """Fixture that provides a TranslationQueueService instance"""
    # Create mock WebSocket
    mock_websocket = AsyncMock()
    
    # Create and initialize the service
    service = TranslationQueueService(mock_websocket)
    
    # Let it initialize
    await asyncio.sleep(0.1)
    
    yield service
    
    # Clean up
    await service.close()

@pytest.mark.asyncio
async def test_items_are_processed_in_order():
    """Test that items are processed in the correct order"""
    # Create mock WebSocket
    mock_websocket = AsyncMock()
    
    # Create the service
    service = TranslationQueueService(mock_websocket)
    
    try:
        # Create test items
        test_items = [
            "1. This is the first test sentence.",
            "2. This is the second test sentence.",
            "3. This is the third test sentence."
        ]
        
        # Add items to the queue
        for i, text in enumerate(test_items):
            await service.enqueue_item(text, f"speaker-{i+1}", f"TEST-{i+1}")
        
        # Wait for all items to be processed
        await service.translation_queue.join()
        
        # Verify results
        assert len(service.processed_items) == 3
        
        # Check that translations were created in the right order
        translations = [item.get("translation", "") for item in service.processed_items]
        expected_translations = [f"TRANSLATED: {sentence}" for sentence in test_items]
        
        # Verify order matches
        assert translations == expected_translations
        
        # Check that websocket.send was called for each item
        assert mock_websocket.send.call_count == 3
    finally:
        # Clean up
        await service.close()

@pytest.mark.asyncio
async def test_enqueue_and_process_single_item(queue_service):
    """Test enqueueing and processing a single item"""
    # Arrange
    service = queue_service
    test_text = "Test sentence"
    
    # Act
    task_id = await service.enqueue_item(test_text, "test-speaker", "TEST-ID")
    
    # Wait for processing
    await service.translation_queue.join()
    
    # Assert
    assert len(service.processed_items) == 1
    assert service.processed_items[0]["text"] == test_text
    assert service.processed_items[0]["translation"] == f"TRANSLATED: {test_text}"
    assert service.processed_items[0]["speaker_id"] == "test-speaker"
    
    # Verify websocket was called
    service.websocket.send.assert_called_once_with(f"TRANSLATED: {test_text}")

@pytest.mark.asyncio
async def test_enqueue_multiple_items_simultaneously(queue_service):
    """Test enqueueing multiple items at once"""
    # Arrange
    service = queue_service
    items = ["First item", "Second item", "Third item"]
    
    # Act - enqueue all items at once
    tasks = []
    for i, item in enumerate(items):
        task = asyncio.create_task(
            service.enqueue_item(item, f"speaker-{i}", f"TASK-{i}")
        )
        tasks.append(task)
    
    # Wait for all enqueue operations to complete
    await asyncio.gather(*tasks)
    
    # Wait for processing to complete
    await service.translation_queue.join()
    
    # Assert - all items should be processed in order
    assert len(service.processed_items) == 3
    
    # Check all items were translated
    for i, item in enumerate(items):
        assert any(p["text"] == item for p in service.processed_items)
        assert any(p["translation"] == f"TRANSLATED: {item}" for p in service.processed_items) 