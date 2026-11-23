import os, re, hashlib, pathlib
from dotenv import load_dotenv
from openai import OpenAI,AsyncOpenAI            
from fastapi import FastAPI
from fastapi.responses import StreamingResponse,FileResponse
from pydantic import BaseModel
import asyncio,time,random,uvicorn,chromadb,docx,logging
from fastapi.middleware.cors import CORSMiddleware
from openai import BadRequestError, AuthenticationError, PermissionDeniedError, NotFoundError, RateLimitError, InternalServerError
from openai import APIError, APIConnectionError, APITimeoutError, APIStatusError

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
embed_client = AsyncOpenAI(
    api_key=os.getenv("qianwen_api"),
    base_url=os.getenv("qianwen_url"),
)
chat_client=AsyncOpenAI(
    api_key=os.getenv("ds_api"),
    base_url=os.getenv("ds_url"),
)

EMBED_MODEL = "text-embedding-v4"
CHAT_MODEL  = "deepseek-flash"        
BASE_DIR=pathlib.Path(__file__).resolve().parent
DB_PATH     = BASE_DIR/"spilt_law_db"
COLL_NAME   = "spilt_law_collection"
DOCX_PATH   = pathlib.Path(__file__).resolve().parent / "中华人民共和国消费者权益保护法_20131025.docx"


logger=logging.getLogger()

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
async def embed_texts(texts, batch_size=10):
    vecs = []
    for i in range(0, len(texts), batch_size):
        resp =await embed_client.embeddings.create(model=EMBED_MODEL, input=texts[i:i+batch_size])
        # 按 index 排序，防止服务端乱序
        vecs += [d.embedding for d in sorted(resp.data, key=lambda x: x.index)]
    return vecs


def get_collection():
    # 关键：空间用 cosine，和"语义相似"的直觉一致（默认 l2）
    return chromadb.PersistentClient(path=DB_PATH).get_or_create_collection(
        name=COLL_NAME, metadata={"hnsw:space": "cosine"}
    )


# ---------- 4. 建库（幂等：用内容 hash 做 id，重复跑不会重复插入）----------
async def build_kb():
    coll = get_collection()
    paras  = load_paragraphs(DOCX_PATH)
    chunks = split_chunks(paras)
    source = DOCX_PATH.name
    

    ids = [hashlib.md5((source + c).encode()).hexdigest() for c in chunks]
    exist = set(coll.get(ids=ids)["ids"])          # 只插入不存在的
    todo  = [(i, c) for i, c in zip(ids, chunks) if i not in exist]
    if not todo:
        print(f"[KB] 已是最新，共 {coll.count()} 块"); return coll

    vecs = await embed_texts([c for _, c in todo])
    coll.add(
        ids=[i for i, _ in todo],
        documents=[c for _, c in todo],
        embeddings=vecs,
        metadatas=[{"source": source, "char_len": len(c)} for _, c in todo],
    )
    print(f"[KB] 新增 {len(todo)} 块，总计 {coll.count()} 块")
    return coll


# ---------- 5. 检索 ----------
async def retrieve(question, k=4, max_dist=0.6):
    coll = get_collection()
    qv = (await embed_texts([question]))[0]
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
continue_prompt=(
    "请紧接着上面 assistant 已经写的内容继续，"
    "不要重复已写内容，不要重新开头，直接输出后续内容。"
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
    
def sse(event: str, data: str) -> str:
    return f"event: {event}\ndata: {data}\n\n"

def strip_overlap(first_resp, new_resp):
    # 从长到短试：已写的最后 k 个字 == 续写开头 k 个字？
    for k in range(min(len(first_resp), len(new_resp), 30), 0, -1):
        if first_resp.endswith(new_resp[:k]):
            return new_resp[k:]      # 砍掉重叠部分
    return new_resp 

async def ask_stream(question, k=4):
    retry=0
    continue_times=0
    max_retrys=2
    collected=[]
    max_continue_char=2000
    max_continue_times=1
    try:
        hits =await retrieve(question, k=k)
        base_messages = build_messages(question, hits)
    except Exception as e:
        logger.exception("检索失败")
        yield sse("error", f"检索失败: {e}")
        return
    while True:
            resp=None
            try:
                if collected:
                    messages=base_messages+[
                        {"role": "assistant", "content": "".join(collected)},
                        {"role": "user",      "content": continue_prompt}]
                else:
                    messages=base_messages
                    base_messages=base_messages
                first_chunk=True
                resp = await chat_client.chat.completions.create(
                            model=CHAT_MODEL,
                            messages=messages,
                            temperature=0.05,
                            stream=True,
                            timeout=25
                        )
                async for chunks in resp:
                        new_reps_firChunk = chunks.choices[0].delta.content if chunks.choices else None
                        if not new_reps_firChunk:
                            continue
                        if collected and first_chunk:
                            new_reps_firChunk=strip_overlap(first_resp="".join(collected),new_resp=new_reps_firChunk)
                            first_chunk=False
                            if not new_reps_firChunk:
                                continue
                        collected.append(new_reps_firChunk)
                        yield sse("token",new_reps_firChunk)
                yield sse("done","")
                return 
            
            except (APITimeoutError,APIConnectionError,RateLimitError,InternalServerError):
                if collected:
                        # 续写
                    yield sse("truncated","流式输出中断")
                    if len("".join(collected)) > max_continue_char:      # 写太多了，续写不划算
                        yield sse("truncated", "流式中断，已保留部分内容")
                        return
                    if continue_times>= max_continue_times:                  # 续写次数用完了（只续 1 次）
                        yield sse("truncated", "续写次数耗尽，已保留部分内容")
                        return
                    continue_times+=1
                    yield sse("retry","正在修复续写中")
                    continue

                    # 无手包，直接全部新写
                else:
                    retry+=1
                    delay=min(2**retry,10)+random.uniform(0,1)
                    await asyncio.sleep(delay)
                    if(retry>=max_retrys):
                        yield f"\nevent: error\ndata: LLM流式请求超时/连接失败\n\n"
                        yield f"\n已经达到最大重试次数"
                        return
    
            except AuthenticationError:
                yield f"\nevent: error\ndata: API Key无效\n\n"
                return
            except APIStatusError as e:
                yield f"\nevent: error\ndata: 服务返回异常，code={e.status_code}, req_id={e.request_id}\n\n"
                yield f"{e}"
                return
            except APIError as e:
                yield f"\nevent: error\ndata: 其他OpenAI SDK异常：{e}\n\n"
                return
            finally:
                if resp is not None:
                    try:
                        await resp.close()
                    except Exception as e:
                        logger.error(e)

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
    asyncio.run(build_kb()) # 启动时确保库是建好的
    print("启动服务，请打开浏览器访问: http://127.0.0.1:8080/spilt_RAG/website.html")
    uvicorn.run(app, host="127.0.0.1", port=8000)