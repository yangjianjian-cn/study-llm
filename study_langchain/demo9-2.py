from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_experimental.tabular_synthetic_data.openai import create_openai_data_generator
from langchain_experimental.tabular_synthetic_data.prompts import SYNTHETIC_FEW_SHOT_PREFIX, SYNTHETIC_FEW_SHOT_SUFFIX
from pydantic import BaseModel

# 生成一些结构化的数据： 5个步骤
# 1、定义数据模型
from study_langchain import devagi_client


# 聊天机器人案例
# 创建模型
class MedicalBilling(BaseModel):
    patient_id: int  # 患者ID，整数类型
    patient_name: str  # 患者姓名，字符串类型
    diagnosis_code: str  # 诊断代码，字符串类型
    procedure_code: str  # 程序代码，字符串类型
    total_charge: float  # 总费用，浮点数类型
    insurance_claim_amount: float  # 保险索赔金额，浮点数类型


# 2、 提供一些样例数据，给AI
examples = [
    {
        "example": "Patient ID: 123456, Patient Name: 张娜, Diagnosis Code: J20.9, Procedure Code: 99203, Total Charge: $500, Insurance Claim Amount: $350"
    },
    {
        "example": "Patient ID: 789012, Patient Name: 王兴鹏, Diagnosis Code: M54.5, Procedure Code: 99213, Total Charge: $150, Insurance Claim Amount: $120"
    },
    {
        "example": "Patient ID: 345678, Patient Name: 刘晓辉, Diagnosis Code: E11.9, Procedure Code: 99214, Total Charge: $300, Insurance Claim Amount: $250"
    },
]

# 3、创建一个提示模板， 用来指导AI生成符合规定的数据
openai_template = PromptTemplate(input_variables=['struc_example'], template="{example}")

prompt_template = FewShotPromptTemplate(
    prefix=SYNTHETIC_FEW_SHOT_PREFIX,
    suffix=SYNTHETIC_FEW_SHOT_SUFFIX,
    examples=examples,
    example_prompt=openai_template,
    input_variables=['subject', 'extra']
)

# 4、创建一个结构化数据的生成器
generator = create_openai_data_generator(
    output_schema=MedicalBilling,  # 指定输出数据的格式
    llm=devagi_client,
    prompt=prompt_template
)

# 5、调用生成器
result = generator.generate(
    subject='医疗账单',  # 指定生成数据的主题
    extra='医疗总费用呈现正态分布，最小的总费用为1000，名字可以是随机的，最好使用比较生僻的人名',  # 额外的一些指导信息
    runs=1  # 指定生成数据的数量
)
print(result)
