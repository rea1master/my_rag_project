import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter # 修正导入
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough,RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import Docx2txtLoader
import chromadb
import pathlib
load_dotenv()

api=os.getenv("qianwen_api")
url=os.getenv("qianwen_url")


llm=ChatOpenAI(
    model="qwen3.7-plus",
    temperature=0.05,
    api_key=api,
    base_url=url
)
embedding=OpenAIEmbeddings(
    model="text-embedding-v4",
    api_key=api,
    base_url=url,
    check_embedding_ctx_length=False,
    chunk_size=10
)

path_base=pathlib.Path(__file__).resolve().parent
docx_path=path_base/"中华人民共和国消费者权益保护法_20131025.docx"

coll_name="law_collection"
persistent_client = chromadb.PersistentClient(path="law_db")
collection = persistent_client.get_or_create_collection("law_collection")
if(collection.count()==0):
    print("向量库未建立")
    loader = Docx2txtLoader(str(docx_path))
    docs=loader.load()
    splitter=RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=70
    )
    splite=splitter.split_documents(docs)
    vectorStore=Chroma.from_documents(
        documents=splite,
        client=persistent_client,
        collection_name=coll_name,
        embedding=embedding,
    )
else:
    print("向量库已存在")
    vectorStore=Chroma(
        embedding_function=embedding,
        client=persistent_client,
        collection_name=coll_name
    )

retriever=vectorStore.as_retriever(search_kwargs={"k":4})

def format_doc(docs):
    seen=set()
    unique=[]
    for d in docs:
        if d.page_content not in seen:
            seen.add(d.page_content)
            unique.append(d)
    return "\n\n".join((d.page_content)for d in unique)

prompt=ChatPromptTemplate(
    [
        ("system","按照文档简洁明了输出信息,补充,额外的必要的说明,禁止编造\n\n{context}"),
        ("human","{question}")
    ]
)

def pre_retri(question):
    docs=retriever.invoke(question)
    return {
        "context":format_doc(docs),
        "docs":docs,
        "question":question
    }

rag_chain=(
    pre_retri|
    RunnableParallel({
        "answer": prompt|llm|StrOutputParser(),
        "docs": lambda x:x["docs"]
    })
)

query="消费者有哪些权利?"
res=rag_chain.invoke(query)
print(res["answer"])
print("\n依据的原文:")
if(len(res["docs"]) !=0):
    for doc in res["docs"]:
        print("---")
        print(doc.page_content)
        print("来源:",doc.metadata.get("source"))
        print("下一块")
else:print("原文档无相关内容,大模型根据自有知识回答")
