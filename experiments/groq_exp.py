import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# 1. 创建 Groq 客户端
client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

def translate(text: str) -> str:
    """
    调用 Groq 对话接口来翻译给定文本，并返回翻译结果。
    """
    # 2. 构造 Prompt
    content = f"""
请帮我翻译下面<START>/<END>中间的内容，然后把翻译结果也放在response的<START>/<END>中间.

<START>
{text}
<END>
"""

    # 3. 调用对话接口
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": content
            }
        ],
        model="llama-3.3-70b-versatile",  # 替换成你实际使用的模型
    )

    # print(chat_completion)

    # 4. 获取模型返回的完整响应文本
    response_text = chat_completion.choices[0].message.content

    # 5. 解析 <START> 和 <END> 之间的翻译内容
    start_tag = "<START>"
    end_tag = "<END>"
    start_idx = response_text.find(start_tag)
    end_idx = response_text.find(end_tag)

    if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
        # 提取并去除首尾空格
        translation = response_text[start_idx + len(start_tag):end_idx].strip()
        return translation
    else:
        # 如果未找到预期的标签，则直接返回整个响应
        return response_text


if __name__ == "__main__":
    # 生成一些测试文本
    test_texts = [
        "This is a dog. It is friendly and always ready to play.",
        "Technology has rapidly advanced over the past decade, changing the way we live.",
        "The quick brown fox jumps over the lazy dog near the river bank.",
        "Artificial intelligence is transforming industries around the world.",
        "Music has a unique ability to evoke emotions and bring people together.",
        "Traveling opens up new perspectives and fosters cultural understanding.",
        "A balanced diet and regular exercise are key to a healthy lifestyle.",
        "History teaches us valuable lessons about our past mistakes and successes.",
        "Nature offers a serene environment to relax and recharge one's mind.",
        "Learning new languages can broaden your horizons and enhance cognitive abilities.",
        "Climate change is one of the most pressing issues of our time.",
        "The world of finance is constantly evolving with new technologies.",
        "Education is the cornerstone of a progressive society.",
        "Innovations in medicine have dramatically improved life expectancy.",
        "The art of storytelling has captivated humans for centuries.",
        "Space exploration fuels our curiosity about the universe.",
        "Environmental conservation is essential for sustaining life on Earth.",
        "Cultural diversity enriches our communities and promotes understanding.",
        "Digital transformation is reshaping the global economy.",
        "Entrepreneurship drives innovation and creates economic opportunities.",
        "In today's fast-paced technological landscape, it is imperative that companies continuously innovate and adapt their strategies to remain competitive in a global market that is increasingly interconnected and data-driven.",
        "Despite the numerous challenges posed by climate change and environmental degradation, researchers and policymakers around the world are working collaboratively to develop sustainable solutions that balance economic growth with ecological preservation.",
        "The rapid advancement of artificial intelligence and machine learning technologies has revolutionized industries ranging from healthcare and finance to transportation and entertainment, prompting a fundamental shift in how we approach problem-solving and decision-making.",
        "In an era where digital transformation is reshaping the business landscape, organizations must invest in robust cybersecurity measures and innovative data management strategies to protect sensitive information and maintain consumer trust.",
        "Education systems around the globe are evolving to better prepare students for the demands of the modern workforce, incorporating new methodologies and technologies that emphasize critical thinking, creativity, and lifelong learning.",
        "Cultural diversity and the exchange of ideas across international borders have long been recognized as key drivers of progress, fostering innovation and mutual understanding in a world that is becoming increasingly polarized.",
        "The integration of renewable energy sources into national power grids is not only crucial for reducing carbon emissions, but also for ensuring long-term energy security and stimulating economic growth in emerging markets.",
        "As public health challenges continue to emerge, it is essential for governments and private organizations to collaborate on developing effective strategies that enhance healthcare delivery, improve access to quality medical services, and promote overall community well-being.",
        "The complexities of modern urban development require a holistic approach that takes into account factors such as infrastructure, transportation, environmental sustainability, and social equity in order to create livable and resilient cities.",
        "Innovative research in the fields of biotechnology and genetics holds the promise of groundbreaking medical treatments and therapies that could fundamentally alter our understanding of human health and disease prevention."
    ]

    for i, text in enumerate(test_texts, start=1):
        translation = translate(text)
        print(f"测试段落 {i}: {text} 翻译结果:\n{translation}\n{'-' * 50}")