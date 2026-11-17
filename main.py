import os, re, hashlib, pathlib
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
import docx              
from fastapi import FastAPI
from fastapi.responses import StreamingResponse,FileResponse
from pydantic import BaseModel
import asyncio
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

class QueryModel(BaseModel):
    question:str

app=FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv()
client = OpenAI(
    api_key=os.getenv("qianwen_api"),
    base_url=os.getenv("qianwen_url"),
)

EMBED_MODEL = "text-embedding-v4"
CHAT_MODEL  = "qwen3.7-plus"        
DB_PATH     = "spilt_law_db"
COLL_NAME   = "spilt_law_collection"
DOCX_PATH   = pathlib.Path(__file__).resolve().parent / "中华人民共和国消费者权益保护法_20131025.docx"


# ---------- 1. 加载 ----------
def load_paragraphs(path):
    d = docx.Document(str(path))
    return [t for p in d.paragraphs if(t:= p.text.strip())]


# # ---------- 2. 切分（法律文本按"条"切，超长再滑窗）----------
ARTICLE_RE = re.compile(r"^第[一二三四五六七八九十百零〇\d]+条")

def _sliding_split(text, max_chars, overlap):
    """按句号/分号/换行切句，再滑窗拼到 max_chars"""
    sents = re.split(r"(?<=[。；！？\n])", text)
    out, buf = [], ""
    for s in sents:
        if buf and len(buf) + len(s) > max_chars:
            out.append(buf)
            buf = buf[-overlap:] if overlap else ""    #保留尾部 overlap 做上下文
        buf += s
    if buf.strip():
        out.append(buf)
    return out

def split_chunks(paras, max_chars=500, overlap=70):
    # 先按"第X条"聚合成块，保证一个法条尽量不被切断
    blocks, cur = [], []
    for p in paras:
        if ARTICLE_RE.match(p) and cur:
            blocks.append("\n".join(cur)); cur = [p]
        else:
            cur.append(p)
    if cur:
        blocks.append("\n".join(cur))

    chunks = []
    for b in blocks:
        chunks += [b] if len(b) <= max_chars else _sliding_split(b, max_chars, overlap)
    return chunks


# ---------- 3. 向量化（DashScope 批量上限 10）----------
def embed_texts(texts, batch_size=10):
    vecs = []
    for i in range(0, len(texts), batch_size):
        resp = client.embeddings.create(model=EMBED_MODEL, input=texts[i:i+batch_size])
        # 按 index 排序，防止服务端乱序
        vecs += [d.embedding for d in sorted(resp.data, key=lambda x: x.index)]
    return vecs


def get_collection():
    # 关键：空间用 cosine，和"语义相似"的直觉一致（默认 l2）
    return chromadb.PersistentClient(path=DB_PATH).get_or_create_collection(
        name=COLL_NAME, metadata={"hnsw:space": "cosine"}
    )


# ---------- 4. 建库（幂等：用内容 hash 做 id，重复跑不会重复插入）----------
def build_kb():
    coll = get_collection()
    paras  = load_paragraphs(DOCX_PATH)
    chunks = split_chunks(paras)
    source = DOCX_PATH.name
    

    ids = [hashlib.md5((source + c).encode()).hexdigest() for c in chunks]
    exist = set(coll.get(ids=ids)["ids"])          # 只插入不存在的
    todo  = [(i, c) for i, c in zip(ids, chunks) if i not in exist]
    if not todo:
        print(f"[KB] 已是最新，共 {coll.count()} 块"); return coll

    vecs = embed_texts([c for _, c in todo])
    coll.add(
        ids=[i for i, _ in todo],
        documents=[c for _, c in todo],
        embeddings=vecs,
        metadatas=[{"source": source, "char_len": len(c)} for _, c in todo],
    )
    print(f"[KB] 新增 {len(todo)} 块，总计 {coll.count()} 块")
    return coll


# ---------- 5. 检索 ----------
def retrieve(question, k=4, max_dist=0.6):
    coll = get_collection()
    qv = embed_texts([question])[0]
    r = coll.query(
        query_embeddings=[qv], n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    hits = list(zip(r["documents"][0], r["metadatas"][0], r["distances"][0]))
    # 距离阈值过滤：太不相关的不给模型，避免它硬编
    return [h for h in hits if h[2] <= max_dist]


# ---------- 6. 拼 prompt + 调 LLM ----------
SYSTEM = (
    "你是《中华人民共和国消费者权益保护法》问答助手。"
    "只能依据【参考资料】回答；资料不足时明确说\"资料中未涉及\"，禁止编造。"
    "回答末尾用 [片段N] 标注依据。"
)

def build_messages(question, hits):
    if not hits:
        return [
            {"role": "system", "content": "资料中未检索到相关内容，请如实告知用户。"},
            {"role": "user", "content": question},
        ]
    ctx = "\n\n".join(f"[片段{i}] {doc}" for i, (doc, _, _) in enumerate(hits, 1))
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"【参考资料】\n{ctx}\n\n【问题】{question}"},
    ]
    

def ask_stream(question, k=4):
    hits = retrieve(question, k=k)
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=build_messages(question, hits),
        temperature=0.05,
        stream=True
    )
    for chunks in resp:
        if chunks.choices and chunks.choices[0].delta.content:
            yield chunks.choices[0].delta.content

@app.post("/chat_stream")
async def chat_stream(request: QueryModel):
    """接收前端提问，返回流式响应"""
    return StreamingResponse(ask_stream(request.question), media_type="text/event-stream")

# ================= 新增：返回前端页面 =================
@app.get("/")
async def get_index():
    # 直接返回同目录下的 index.html 文件
    return FileResponse("website.html")

if __name__ == "__main__":
    print("正在检查/构建知识库...")
    build_kb() # 启动时确保库是建好的
    print("启动服务，请打开浏览器访问: http://127.0.0.1:8080/spilt_RAG/website.html")
    uvicorn.run(app, host="127.0.0.1", port=8000)