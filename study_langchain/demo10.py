from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from study_langchain import devagi_client


class Classification1(BaseModel):
    """
        定义一个Pydantic的数据模型，未来需要根据该类型，完成文本的分类
    """
    # 文本的情感倾向，预期为字符串类型
    sentiment: str = Field(description="文本的情感")

    # 文本的攻击性，预期为1到10的整数
    aggressiveness: int = Field(
        description="描述文本的攻击性，数字越大表示越攻击性"
    )

    # 文本使用的语言，预期为字符串类型
    language: str = Field(description="文本使用的语言")


# 聊天机器人案例
# 创建模型


class Classification2(BaseModel):
    """
        定义一个Pydantic的数据模型，未来需要根据该类型，完成文本的分类
    """
    # 文本的情感倾向，预期为字符串类型
    sentiment: str = Field(..., enum=["happy", "neutral", "sad"], description="文本的情感")

    # 文本的攻击性，预期为1到5的整数
    aggressiveness: int = Field(..., enum=[1, 2, 3, 4, 5], description="描述文本的攻击性，数字越大表示越攻击性")

    # 文本使用的语言，预期为字符串类型
    language: str = Field(..., enum=["spanish", "english", "french", "中文", "italian"], description="文本使用的语言")


# 创建一个用于提取信息的提示模板
tagging_prompt = ChatPromptTemplate.from_template(
    """
    从以下段落中提取所需信息。
    只提取'Classification2'类中提到的属性。
    段落：
    {input}
    """
)

chain = tagging_prompt | devagi_client.with_structured_output(Classification2)

input_text = "中国人民大学的王教授：师德败坏，做出的事情实在让我生气！"
# input_text = "Estoy increiblemente contento de haberte conocido! Creo que seremos muy buenos amigos!"
result: Classification2 = chain.invoke({'input': input_text})
print(result)
# sentiment='negative' aggressiveness=7 language='zh'