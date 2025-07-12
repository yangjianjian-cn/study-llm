from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate

# 创建模型
# 加载我们的文档。我们将使用 WebBaseLoader 来加载博客文章：
from study_langchain import chatOpenAI_client

loader = WebBaseLoader('https://lilianweng.github.io/posts/2023-06-23-agent/')
docs = loader.load()  # 得到整篇文章

# 第一种： Stuff

# Stuff的第一种写法
# chain = load_summarize_chain(model, chain_type='stuff')

# Stuff的第二种写法
# 定义提示
prompt_template = """针对下面的内容，写一个简洁的总结摘要:
"{text}"
简洁的总结摘要:"""
prompt = PromptTemplate.from_template(prompt_template)

llm_chain = LLMChain(llm=chatOpenAI_client, prompt=prompt)
stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name='text')

result = stuff_chain.invoke(docs)
print(result['output_text'])
