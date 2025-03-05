import asyncio
import pytest
from sonara.azure_cog import async_translate


# 使用 pytest 的 monkeypatch 将 translate_text 替换为一个 dummy 实现
def dummy_translate_text(text: str) -> str:
    return f"Dummy translation for: {text}"


@pytest.mark.asyncio
async def test_async_translate(monkeypatch):
    # 替换掉 azure_cog 模块中 translate_text 函数
    monkeypatch.setattr("sonara.azure_cog.translate_text", dummy_translate_text)
    sample_text = "你好世界"
    loop = asyncio.get_running_loop()
    translation = await async_translate(sample_text, loop)
    assert translation == f"Dummy translation for: {sample_text}"
