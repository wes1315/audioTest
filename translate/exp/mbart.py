from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

model_name = "facebook/mbart-large-50-many-to-many-mmt"
tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
model = MBartForConditionalGeneration.from_pretrained(model_name)

# 设置源语言和目标语言
tokenizer.src_lang = "en_XX"
target_lang = "zh_CN"  # 根据需要调整

src_text = """We got some new snacks today.
Hummus and pretzel cups and Pho bowl, beef flavor.
Vegetarians please be aware the new Pho bowl is beef so therefore not vegetarian.
The veggie Pho bowl is out of stock at this time. I will stock that when it is available again."""

encoded = tokenizer(src_text, return_tensors="pt")
generated_tokens = model.generate(**encoded, forced_bos_token_id=tokenizer.lang_code_to_id[target_lang])
translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

print(translation)
