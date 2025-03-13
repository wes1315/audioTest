# 假设存在一个 Ollama Python API，提供 load 和 run 方法
# 这里给出一个伪代码示例

# class OllamaModel:
#     def __init__(self, model_name: str):
#         """
#         初始化模型对象，加载指定的模型。
#         """
#         self.model_name = model_name
#         # 模型加载的过程，实际代码请参考 Ollama 的 Python API 文档
#         self._load_model()

#     def _load_model(self):
#         # 这里模拟模型加载过程
#         print(f"Loading model {self.model_name}...")
#         # 假设加载过程耗时，此处省略具体实现
#         print(f"Model {self.model_name} loaded.")

#     def run(self, prompt: str) -> str:
#         """
#         模拟模型推理方法，实际调用时请替换为对应的接口。
#         这里假设返回的文本已经包含 <START> 和 <END> 标签，
#         例如：
#             "<START>\nYou handed your thoughts to the TV, your connections to the phone, your legs to the car, and your health to pills.\n<END>"
#         """
#         # 实际使用时调用模型推理接口，传入 prompt，返回模型输出
#         # 下面为模拟返回值
#         simulated_output = (
#             "<START>\n"
#             "You handed your thoughts to the TV, your connections to the phone, your legs to the car, and your health to pills.\n"
#             "<END>"
#         )
#         return simulated_output


def load_model(model_name: str = "deepseek-r1:14b") -> OllamaModel:
    """
    加载指定名称的 Ollama 模型，并返回模型对象。

    参数:
      model_name: 模型名称，默认为 deepseek-r1:14b

    返回:
      已加载的模型对象
    """
    return OllamaModel(model_name)


def translate_cn_to_en(loaded_model: OllamaModel, text: str) -> str:
    """
    使用已加载的 Ollama deepseek-r1:14b 模型将中文文本翻译成英文，
    并用 <START> 和 <END> 包裹翻译结果，同时提取标签中的实际翻译内容。

    参数:
      loaded_model: 已加载的 deepseek 模型对象，提供 run() 方法执行推理
      text: 需要翻译的中文文本

    返回:
      翻译后的英文字符串（提取标签内的内容）
    """
    # 构造提示信息，要求模型翻译并用 <START> 和 <END> 包裹结果
    prompt = f"""Please help me translate the following paragraph into English and wrap it with <START> and <END>:

<START>
{text}
<END>
"""
    # 调用模型的推理方法
    output = loaded_model.run(prompt)
    
    # 提取输出中 <START> 和 <END> 之间的内容
    start_marker = "<START>"
    end_marker = "<END>"
    start_index = output.find(start_marker)
    end_index = output.find(end_marker, start_index + len(start_marker))
    
    if start_index != -1 and end_index != -1:
        # 提取出实际翻译的内容
        translated_text = output[start_index + len(start_marker):end_index].strip()
        return translated_text
    else:
        # 如果标签未找到，返回整个输出文本
        return output.strip()


# 示例调用
if __name__ == "__main__":
    # 1. 加载模型
    print("开始加载模型")
    loaded_model = load_model("deepseek-r1:14b")
    print("模型加载完成")
    
    chinese_text = "你把思考交给了电视，把联系交给了手机，把双腿交给了汽车，把健康交给了药丸。"
    translated_result = translate_cn_to_en(loaded_model, chinese_text)
    
    # 4. 输出翻译结果
    print("翻译结果：")
    print(translated_result)
