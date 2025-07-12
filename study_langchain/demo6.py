from operator import itemgetter

from langchain.chains.sql_database.query import create_sql_query_chain
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

from study_langchain import devagi_client, USERNAME, PASSWORD, HOSTNAME, PORT, DATABASE

# 聊天机器人案例
# 创建模型

# sqlalchemy 初始化MySQL数据库的连接
MYSQL_URI = 'mysql+mysqldb://{}:{}@{}:{}/{}?charset=utf8mb4'.format(USERNAME, PASSWORD, HOSTNAME, PORT, DATABASE)

db = SQLDatabase.from_uri(MYSQL_URI)

# 测试连接是否成功
# print(db.get_usable_table_names())
# print(db.run('select * from chip_dict limit 10;'))

# 直接使用大模型和数据库整合, 只能根据你的问题生成SQL
# 初始化生成SQL的chain
test_chain = create_sql_query_chain(devagi_client, db)
# resp = test_chain.invoke({'question': '请问：chip_dict表中有多少条数据？'})
# print(resp)

answer_prompt = PromptTemplate.from_template(
    """给定以下用户问题、SQL语句和SQL执行后的结果，回答用户问题。
    Question: {question}
    SQL Query: {query}
    SQL Result: {result}
    回答: """
)
# 创建一个执行sql语句的工具
execute_sql_tool = QuerySQLDataBaseTool(db=db)


def clean_sql(raw_sql: str) -> str:
    # 去除 Markdown 标记
    cleaned = raw_sql.replace("```sql", "").replace("```", "")
    # 去除前后空白
    cleaned = cleaned.strip()
    return cleaned


# 1、生成SQL，2、执行SQL
# 2、模板
chain = (RunnablePassthrough.assign(query=test_chain | clean_sql).assign(result=itemgetter('query') | execute_sql_tool)
         | answer_prompt
         | devagi_client
         | StrOutputParser()
         )

rep = chain.invoke(input={'question': 'chip_dict表中有多少条数据'})
print(rep)
