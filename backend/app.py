import os
from pathlib import Path
from typing import Any

from chromadb import PersistentClient
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from litellm import completion
from pydantic import BaseModel
from pypdf import PdfReader

ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")


class IndexRequest(BaseModel):
    pdf_dir: str | None = None
    reset: bool = True


class AskRequest(BaseModel):
    question: str
    top_k: int | None = None


app = FastAPI(title="RAGlocal Backend", version="1.0.0")


def get_env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def get_settings() -> dict[str, Any]:
    return {
        "pdf_dir": os.getenv("RAGLOCAL_PDF_DIR", ""),
        "db_dir": os.getenv("RAGLOCAL_DB_DIR", "raglocal_db"),
        "collection_name": os.getenv("RAGLOCAL_COLLECTION", "raglocal_docs"),
        "embed_model": os.getenv("RAGLOCAL_EMBED_MODEL", "all-MiniLM-L6-v2"),
        "llm_model": os.getenv("RAGLOCAL_MODEL", "groq/llama-3.3-70b-versatile"),
        "api_key": os.getenv("GROQ_API_KEY", ""),
        "top_k": get_env_int("RAGLOCAL_TOP_K", 5),
        "chunk_size": get_env_int("RAGLOCAL_CHUNK_SIZE", 1000),
        "chunk_overlap": get_env_int("RAGLOCAL_CHUNK_OVERLAP", 150),
    }


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    text = " ".join(text.split())
    if not text:
        return []
    if overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return chunks


def find_pdf_files(pdf_dir: Path) -> list[Path]:
    return sorted([p for p in pdf_dir.rglob("*.pdf") if p.is_file()])


def get_collection(db_dir: str, collection_name: str, embed_model: str):
    embedding_fn = SentenceTransformerEmbeddingFunction(model_name=embed_model)
    client = PersistentClient(path=db_dir)
    return client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )


def index_documents(pdf_dir: str, reset: bool) -> dict[str, Any]:
    settings = get_settings()

    pdf_root = Path(pdf_dir).expanduser().resolve()
    if not pdf_root.exists() or not pdf_root.is_dir():
        raise HTTPException(status_code=400, detail=f"PDF directory not found: {pdf_root}")

    files = find_pdf_files(pdf_root)
    if not files:
        return {
            "ok": True,
            "message": f"No PDF files found in: {pdf_root}",
            "summary": {
                "indexed_chunks": 0,
                "skipped_password_protected": 0,
                "skipped_unreadable": 0,
                "skipped_no_text": 0,
            },
            "logs": [f"No PDF files found in: {pdf_root}"],
        }

    client = PersistentClient(path=settings["db_dir"])
    logs: list[str] = [f"Using PDF directory: {pdf_root}"]

    if reset:
        try:
            client.delete_collection(settings["collection_name"])
            logs.append(f"Reset collection: {settings['collection_name']}")
        except Exception:
            logs.append("Collection not found previously; starting fresh")

    collection = get_collection(
        settings["db_dir"], settings["collection_name"], settings["embed_model"]
    )

    ids: list[str] = []
    docs: list[str] = []
    metas: list[dict] = []

    total_chunks = 0
    skipped_protected = 0
    skipped_unreadable = 0
    skipped_no_text = 0

    logs.append(f"Found {len(files)} PDF file(s). Starting indexing...")

    for pdf_path in files:
        rel_path = str(pdf_path.relative_to(pdf_root))
        try:
            reader = PdfReader(str(pdf_path))
        except Exception as err:
            skipped_unreadable += 1
            logs.append(f"Skipping unreadable PDF: {rel_path} ({err})")
            continue

        if reader.is_encrypted:
            try:
                decrypt_status = reader.decrypt("")
            except Exception:
                decrypt_status = 0
            if decrypt_status == 0:
                skipped_protected += 1
                logs.append(f"Skipping password-protected PDF: {rel_path}")
                continue

        file_chunk_count = 0
        file_text_chars = 0

        for page_idx, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            file_text_chars += len(page_text.strip())
            chunks = chunk_text(
                page_text,
                chunk_size=settings["chunk_size"],
                overlap=settings["chunk_overlap"],
            )
            for chunk_idx, chunk in enumerate(chunks, start=1):
                chunk_id = f"{rel_path}:{page_idx}:{chunk_idx}"
                ids.append(chunk_id)
                docs.append(chunk)
                metas.append({"source": rel_path, "page": page_idx, "chunk": chunk_idx})
                file_chunk_count += 1

        if file_chunk_count == 0:
            skipped_no_text += 1
            logs.append(
                f"No extractable text found in: {rel_path} (possibly scanned/image-only PDF)"
            )
        else:
            logs.append(
                f"Extracted {file_chunk_count} chunks from: {rel_path} (text chars: {file_text_chars})"
            )

        if len(ids) >= 200:
            collection.upsert(ids=ids, documents=docs, metadatas=metas)
            total_chunks += len(ids)
            logs.append(f"Upserted {total_chunks} chunks so far...")
            ids, docs, metas = [], [], []

    if ids:
        collection.upsert(ids=ids, documents=docs, metadatas=metas)
        total_chunks += len(ids)

    return {
        "ok": True,
        "message": "Indexing complete",
        "summary": {
            "indexed_chunks": total_chunks,
            "skipped_password_protected": skipped_protected,
            "skipped_unreadable": skipped_unreadable,
            "skipped_no_text": skipped_no_text,
            "collection": settings["collection_name"],
        },
        "logs": logs,
    }


def query_context(question: str, top_k: int) -> tuple[str, list[str]]:
    settings = get_settings()
    collection = get_collection(
        settings["db_dir"], settings["collection_name"], settings["embed_model"]
    )

    result = collection.query(
        query_texts=[question],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]

    if not documents:
        return "", []

    context_parts: list[str] = []
    citations: list[str] = []
    for idx, (doc, meta, distance) in enumerate(zip(documents, metadatas, distances), start=1):
        source = str(meta.get("source", "unknown"))
        page = str(meta.get("page", "?"))
        cite = f"{source} (page {page})"
        citations.append(cite)
        context_parts.append(
            f"[{idx}] Source: {cite}\nSimilarity distance: {distance:.4f}\nContent:\n{doc}"
        )

    return "\n\n".join(context_parts), list(dict.fromkeys(citations))


def answer_with_llm(question: str, context: str) -> str:
    settings = get_settings()
    api_key = settings["api_key"]
    if not api_key:
        raise HTTPException(status_code=400, detail="GROQ_API_KEY is missing in .env")

    system_prompt = (
        "You are a RAG assistant. Use only the provided context to answer. "
        "If context is insufficient, say so clearly. "
        "Always include source citations in the form: source (page N)."
    )
    user_prompt = (
        f"Question:\n{question}\n\n"
        f"Context:\n{context}\n\n"
        "Respond with:\n"
        "1) Direct answer\n"
        "2) Short explanation\n"
        "3) Citations used"
    )

    response = completion(
        model=settings["llm_model"],
        api_key=api_key,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )

    return str(response["choices"][0]["message"]["content"])


@app.get("/health")
def health() -> dict[str, Any]:
    settings = get_settings()
    return {
        "ok": True,
        "collection": settings["collection_name"],
        "db_dir": settings["db_dir"],
        "pdf_dir": settings["pdf_dir"],
    }


@app.post("/index")
def index_route(req: IndexRequest) -> dict[str, Any]:
    settings = get_settings()
    pdf_dir = req.pdf_dir if req.pdf_dir else settings["pdf_dir"]
    if not pdf_dir:
        raise HTTPException(status_code=400, detail="pdf_dir missing and RAGLOCAL_PDF_DIR not set")
    return index_documents(pdf_dir=pdf_dir, reset=req.reset)


@app.post("/ask")
def ask_route(req: AskRequest) -> dict[str, Any]:
    settings = get_settings()
    top_k = req.top_k if req.top_k is not None else settings["top_k"]

    context, citations = query_context(req.question, top_k=top_k)
    if not context:
        return {
            "ok": True,
            "answer": "No relevant chunks found. Please run indexing or verify document text extraction.",
            "citations": [],
        }

    answer = answer_with_llm(req.question, context)
    return {"ok": True, "answer": answer, "citations": citations}


@app.get("/stats")
def stats() -> dict[str, Any]:
    settings = get_settings()
    collection = get_collection(
        settings["db_dir"], settings["collection_name"], settings["embed_model"]
    )
    result = collection.get(include=["metadatas"])
    metadatas = result.get("metadatas", [])
    sources = sorted({m.get("source", "") for m in metadatas if m})
    return {
        "ok": True,
        "count": collection.count(),
        "unique_sources": len(sources),
        "sources": sources,
    }
