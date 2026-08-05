# rag_api.py
from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager
import re

from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ── Globals ────────────────────────────────────────────────────────────────
rag_chain = None

GROQ_API_KEY = "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"
QDRANT_PATH  = "C:\\model_sh1\\qdrant_db"   # ← your local path

# ── Startup: build the chain once ─────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_chain

    print("⏳ Loading embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
    )

    print("⏳ Connecting to Qdrant...")
    client = QdrantClient(path=QDRANT_PATH)
    vectorstore = QdrantVectorStore(
        client=client,
        collection_name="arabic_medical_qa",
        embedding=embeddings,
    )
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

    print("⏳ Loading reranker...")
    reranker_model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")
    compressor = CrossEncoderReranker(model=reranker_model, top_n=4)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever,
    )

    llm = ChatGroq(api_key=GROQ_API_KEY, model="llama-3.3-70b-versatile", temperature=0.7)

    template = """<|im_start|>system
أنت مساعد طبي عربي متخصص. قواعد صارمة:
- استخدم العربية الفصحى فقط في إجابتك
- إذا وجدت نصاً بغير العربية في السياق، تجاهله تماماً ولا تترجمه
- استخدم فقط المعلومات الطبية العربية من السياق
- لا تكرر الجمل ولا تضف معلومات خارج السياق
- أنهِ إجابتك بـ: "يُنصح دائماً باستشارة الطبيب المختص."
<|im_end|>
<|im_start|>user
السياق الطبي:
{context}
السؤال: {question}
<|im_end|>
<|im_start|>assistant
"""
    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs):
        cleaned = []
        for doc in docs:
            text = re.sub(r'[\u4e00-\u9fff\u3040-\u30ff]+', '', doc.page_content).strip()
            if len(text) > 30:
                cleaned.append(text)
        return "\n\n---\n\n".join(cleaned)

    rag_chain = (
        {"context": compression_retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print("✅ RAG chain ready!")
    yield  # app runs here


# ── App ────────────────────────────────────────────────────────────────────
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    messages: list[dict]   # [{"role": "user"|"assistant", "content": "..."}]

class ChatResponse(BaseModel):
    reply: str

@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    # Use only the last user message as the question for RAG
    question = next(
        (m["content"] for m in reversed(req.messages) if m["role"] == "user"),
        ""
    )
    raw = rag_chain.invoke(question)
    raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL)
    bad_tokens = ["Human:", "Assistant:", "السؤال:", "الإجابة:",
                  "<|im_start|>", "<|im_end|>", "assistant", "user"]
    for b in bad_tokens:
        raw = raw.replace(b, "")
    raw = re.sub(r'[\u4e00-\u9fff\u3040-\u30ff]+', '', raw)
    raw = re.sub(r'\n{3,}', '\n\n', raw).strip()
    return ChatResponse(reply=raw)