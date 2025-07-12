from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory

# 定义提示模板
from study_langchain import chatOpenAI_client

prompt_template = ChatPromptTemplate.from_messages([
    ('system', '你是一个乐于助人的助手。用{language}尽你所能回答所有问题。'),
    MessagesPlaceholder(variable_name='my_msg')
])

# 得到链
chain = prompt_template | chatOpenAI_client

# 保存聊天的历史记录
store = {}  # 所有用户的聊天记录都保存到store。key: sessionId,value: 历史聊天记录对象ChatMessageHistory


# 此函数预期将接收一个session_id并返回一个消息历史记录对象。
def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


do_message = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key='my_msg'  # 每次聊天时候发送msg的key
)

config = {'configurable': {'session_id': '000000001'}}  # 给当前会话定义一个sessionId

# 第一轮
resp1 = do_message.invoke(
    {
        'my_msg': [HumanMessage(content='你好啊！ 我是衣绣云')],
        'language': '中文'
    },
    config=config
)
print(resp1.content)

# 第二轮
resp2 = do_message.invoke(
    {
        'my_msg': [HumanMessage(content='请问：我的名字是什么？')],
        'language': '中文'
    },
    config=config
)
print(resp2.content)

# 第3轮： 返回的数据是流式的
config = {'configurable': {'session_id': '000000002'}}  # 给当前会话定义一个sessionId,新开启一个会话
for resp in do_message.stream(
        {
            'my_msg': [HumanMessage(content='请给我讲一个笑话？')],
            'language': 'English'
        },
        config=config):
    # 每一次resp都是一个token
    print(resp.content, end='-')

print(store)