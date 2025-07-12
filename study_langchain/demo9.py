from langchain_experimental.synthetic_data import create_data_generation_chain

# 创建链
from study_langchain import chatOpenAI_client

# 聊天机器人案例
# 创建模型

chain = create_data_generation_chain(chatOpenAI_client)

# 生成数据
# result = chain(  # 给于一些关键词， 随机生成一句话
#     {
#         "fields": ['蓝色', '黄色'],
#         "preferences": {}
#     }
# )

result = chain(  # 给于一些关键词， 随机生成一句话
    {
        "fields": {"颜色": ['蓝色', '黄色']},
        "preferences": {"style": "让它像诗歌一样。"}
    }
)
print(result)

