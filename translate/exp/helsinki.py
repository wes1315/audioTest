from transformers import pipeline

# 加载中译英模型
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-zh")

text = """
We got some new snacks today.  
Hummus and pretzel cups and Pho bowl, beef flavor.
Vegetarians please be aware the new Pho bowl is beef so therefore not vegetarian.
The veggie Pho bowl is out of stock at this time.  I will stock that when it is available again."""
# 翻译示例文本
result = translator(text)
print(result[0]['translation_text'])
