from ollama import chat
from ollama import ChatResponse

text = """据路透社和美媒当地时间2月14日报道，美国总统特朗普及其当前最重要顾问马斯克发起大幅削减美国官僚机构的运动，其规模于当天持续扩大——从管理联邦土地到照顾退伍军人，从野火预防到医学研究，9500多名从事着各种事务的政府工作人员遭到了解雇，涵盖内政部、能源部、退伍军人事务部、农业部、卫生与公共服务部等各个部门。

“Axios新闻网”指出，如此大规模的解雇行动可谓“前所未有”，并很可能会在未来许多年里极大地重塑美国联邦政府的工作方式，或者造成缺乏工作岗位的局面。
"""

prompt = f"""
Please help me translate the following paragraph into English and wrap it with <START> and <END>:

<START>
{text}
<END>
Please translate the following text into English. Do not include any chain-of-thought or reasoning. Just give the final translation.
"""


response: ChatResponse = chat(model='deepseek-r1:14b', messages=[
  {
    'role': 'user',
    'content': prompt,
  },
])

print(response['message']['content'])
