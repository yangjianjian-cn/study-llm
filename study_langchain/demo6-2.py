from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# sqlalchemy 初始化MySQL数据库的连接
from langgraph.prebuilt.chat_agent_executor import create_tool_calling_executor

from study_langchain import chatOpenAI_client
from study_langchain import devagi_client, USERNAME, PASSWORD, HOSTNAME, PORT, DATABASE

# 聊天机器人案例
# 创建模型
# mysqlclient驱动URL
MYSQL_URI = 'mysql+mysqldb://{}:{}@{}:{}/{}?charset=utf8mb4'.format(USERNAME, PASSWORD, HOSTNAME, PORT, DATABASE)

db = SQLDatabase.from_uri(MYSQL_URI)

# 创建工具
toolkit = SQLDatabaseToolkit(db=db, llm=chatOpenAI_client)
tools = toolkit.get_tools()

# 使用agent完整整个数据库的整合
system_prompt = """
您是一个被设计用来与SQL数据库交互的代理。
给定一个输入问题，创建一个语法正确的SQL语句并执行，然后查看查询结果并返回答案。
除非用户指定了他们想要获得的示例的具体数量，否则始终将SQL查询限制为最多10个结果。
你可以按相关列对结果进行排序，以返回MySQL数据库中最匹配的数据。
您可以使用与数据库交互的工具。在执行查询之前，你必须仔细检查。如果在执行查询时出现错误，请重写查询SQL并重试。
不要对数据库做任何DML语句(插入，更新，删除，删除等)。

首先，你应该查看数据库中的表，看看可以查询什么。
不要跳过这一步。
然后查询最相关的表的模式。
"""
# 构建 PromptTemplate
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="messages"),
])

# 创建 Agent Executor
# 创建代理
# agent_executor = chat_agent_executor.create_tool_calling_executor(devagi_client, tools, system_message)
agent_executor = create_tool_calling_executor(
    model=chatOpenAI_client,  # 你的大模型客户端
    tools=tools,  # SQL 工具列表
    prompt=prompt  # 系统提示
)

# resp = agent_executor.invoke({'messages': [HumanMessage(content='请问：chip_dict表中有多少条数据？')]})
# resp = agent_executor.invoke({'messages': [HumanMessage(content='那种性别的员工人数最多？')]})
resp = agent_executor.invoke({'messages': [HumanMessage(content='chip_dict表有哪些label？')]})

result = resp['messages']
print(result)
print(len(result))
# 最后一个才是真正的答案
print(result[len(result) - 1])
