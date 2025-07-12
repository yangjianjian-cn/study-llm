from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import CharacterTextSplitter
# 创建模型
# 加载我们的文档。我们将使用 WebBaseLoader 来加载博客文章：
from study_langchain import chatOpenAI_client

loader = WebBaseLoader('https://lilianweng.github.io/posts/2023-06-23-agent/')
docs = loader.load()  # 得到整篇文章

# 第三种： Refine
'''
Refine: RefineDocumentsChain 类似于map-reduce：
文档链通过循环遍历输入文档并逐步更新其答案来构建响应。对于每个文档，它将当前文档和最新的中间答案传递给LLM链，以获得新的答案。
'''
# 第一步： 切割阶段
# 每一个小docs为1000个token
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000, chunk_overlap=0)
split_docs = text_splitter.split_documents(docs)

# 指定chain_type为： refine
chain = load_summarize_chain(chatOpenAI_client, chain_type='refine')

result = chain.invoke(split_docs)
print(result['output_text'])
