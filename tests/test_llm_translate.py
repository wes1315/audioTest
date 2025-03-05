from logging import info
import logging
import pytest
from sonara.llm_translate import translate_text

logging.basicConfig(level=logging.DEBUG)


def test_translate_text():
    info("test_translate_text")
    # 示例中文文本
    sample_text = """
Thanks especially to the host of the Munich Security Conference for being able to, to put on such an incredible event or of course, thrilled to be here. We're happy to be here. And you know, one of the things that I wanted to, to talk about today is of course, our shared values. And you know, it's, it's great to be back in Germany as as you heard earlier, I was here last year as United States senator. I saw a foreign minister, excuse me, Foreign Secretary David Lammy and joked that both of us last year had different jobs than we have now, but.
"""
    translation = translate_text(sample_text)
    info("translation: '{}'".format(translation))
    # 确保翻译结果不为空
    assert translation != "", "Translation result should not be empty"
    # 可以增加进一步检查，比如确认翻译结果中没有包含 <START> 和 <END> 标记
    assert "<START>" not in translation and "<END>" not in translation, "Translation should not contain markers"
