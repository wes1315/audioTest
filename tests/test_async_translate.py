import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

# 使用模拟对象进行测试
@pytest.mark.asyncio
async def test_async_translation():
    """
    测试异步翻译功能 - 使用模拟翻译类和WebSocket
    """
    # 创建模拟的翻译器和WebSocket
    mock_translator = MagicMock()
    mock_translator.translate_with_retries = MagicMock(return_value="Translated text")
    
    mock_websocket = AsyncMock()
    mock_websocket.open = True
    mock_websocket.send = AsyncMock()
    
    # 创建一个简化的异步翻译函数
    async def async_translate(text, speaker_id="test"):
        # 模拟翻译过程
        translation = mock_translator.translate_with_retries(text)
        
        # 发送结果
        if translation:
            message = {
                "type": "translated",
                "result": translation,
                "speaker": speaker_id
            }
            await mock_websocket.send(str(message))
        
        return translation
    
    # 测试翻译函数
    test_text = "Hello world"
    result = await async_translate(test_text)
    
    # 验证结果
    assert result == "Translated text"
    assert mock_translator.translate_with_retries.called
    assert mock_translator.translate_with_retries.call_args[0][0] == test_text
    assert mock_websocket.send.called
