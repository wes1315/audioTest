from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def load_model(model_name: str = "meta-llama/Llama-3.2-3B-Instruct"):
    """
    加载指定的 Llama 3.2 3B 模型及其分词器。注意：如果模型是 gated 模型，
    你需要先使用 `huggingface-cli login` 登录，并接受许可协议，然后才能下载模型。
    """
    print(f"Loading model {model_name}...")
    # 如果需要认证，请确保你已经登录，或者在这里传入 use_auth_token=True
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=True)
    print("Model loaded.")
    return tokenizer, model

def translate_cn_to_en(tokenizer, model, text: str) -> str:
    """
    使用模型将中文文本翻译成英文。构造 prompt 要求模型翻译并用 <START> 和 <END> 包裹，
    然后从输出中提取标签之间的内容作为最终翻译结果。
    """
    prompt = f"""
Please help me translate the following paragraph into English and wrap it with <START> and <END>:

<START>
{text}
<END>
Please translate the following text into English. Do not include any chain-of-thought or reasoning. Just give the final translation.
"""
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=200)
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Raw output:", output_text)
    
    # 提取 <START> 和 <END> 标签之间的内容
    start_marker = "<START>"
    end_marker = "<END>"
    start_index = output_text.find(start_marker)
    end_index = output_text.find(end_marker, start_index + len(start_marker))
    
    if start_index != -1 and end_index != -1:
        translated_text = output_text[start_index + len(start_marker):end_index].strip()
        return translated_text
    else:
        return output_text.strip()

if __name__ == "__main__":
    text = """据路透社和美媒当地时间2月14日报道，美国总统特朗普及其当前最重要顾问马斯克发起大幅削减美国官僚机构的运动，其规模于当天持续扩大——从管理联邦土地到照顾退伍军人，从野火预防到医学研究，9500多名从事着各种事务的政府工作人员遭到了解雇，涵盖内政部、能源部、退伍军人事务部、农业部、卫生与公共服务部等各个部门。

“Axios新闻网”指出，如此大规模的解雇行动可谓“前所未有”，并很可能会在未来许多年里极大地重塑美国联邦政府的工作方式，或者造成缺乏工作岗位的局面。
"""
    # 加载模型和分词器
    tokenizer, model = load_model("meta-llama/Llama-3.2-3B-Instruct")
    
    # 调用翻译函数
    translation = translate_cn_to_en(tokenizer, model, text)
    print("翻译结果：")
    print(translation)
