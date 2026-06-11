import argparse
import os
from pathlib import Path

from chromadb import PersistentClient
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from dotenv import load_dotenv
from litellm import completion
from pypdf import PdfReader


def get_env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    text = " ".join(text.split())
    if not text:
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

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


def index_pdfs(
    pdf_dir: str,
    db_dir: str,
    collection_name: str,
    embed_model: str,
    chunk_size: int,
    chunk_overlap: int,
    reset_collection: bool,
) -> None:
    pdf_root = Path(pdf_dir).expanduser().resolve()
    if not pdf_root.exists() or not pdf_root.is_dir():
        raise FileNotFoundError(f"PDF directory not found: {pdf_root}")

    print(f"Using PDF directory: {pdf_root}")

    files = find_pdf_files(pdf_root)
    if not files:
        print(f"No PDF files found in: {pdf_root}")
        return

    client = PersistentClient(path=db_dir)
    if reset_collection:
        try:
            client.delete_collection(collection_name)
            print(f"Reset collection: {collection_name}")
        except Exception:
            # Ignore when collection does not exist yet.
            pass

    collection = get_collection(db_dir, collection_name, embed_model)

    ids: list[str] = []
    docs: list[str] = []
    metas: list[dict] = []

    print(f"Found {len(files)} PDF file(s). Starting indexing...")
    total_chunks = 0
    skipped_protected = 0
    skipped_unreadable = 0
    skipped_no_text = 0

    for pdf_path in files:
        rel_path = str(pdf_path.relative_to(pdf_root))
        try:
            reader = PdfReader(str(pdf_path))
        except Exception as err:
            skipped_unreadable += 1
            print(f"Skipping unreadable PDF: {rel_path} ({err})")
            continue

        if reader.is_encrypted:
            try:
                decrypt_status = reader.decrypt("")
            except Exception:
                decrypt_status = 0

            if decrypt_status == 0:
                skipped_protected += 1
                print(f"Skipping password-protected PDF: {rel_path}")
                continue

        print(f"Reading: {rel_path} ({len(reader.pages)} page(s))")
        file_chunk_count = 0
        file_text_chars = 0

        for page_idx, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            file_text_chars += len(page_text.strip())
            for chunk_idx, chunk in enumerate(
                chunk_text(page_text, chunk_size=chunk_size, overlap=chunk_overlap),
                start=1,
            ):
                chunk_id = f"{rel_path}:{page_idx}:{chunk_idx}"
                ids.append(chunk_id)
                docs.append(chunk)
                file_chunk_count += 1
                metas.append(
                    {
                        "source": rel_path,
                        "page": page_idx,
                        "chunk": chunk_idx,
                    }
                )

        if file_chunk_count == 0:
            skipped_no_text += 1
            print(
                f"No extractable text found in: {rel_path} "
                "(possibly scanned/image-only PDF)"
            )
        else:
            print(
                f"Extracted {file_chunk_count} chunks from: {rel_path} "
                f"(text chars: {file_text_chars})"
            )

        if len(ids) >= 200:
            collection.upsert(ids=ids, documents=docs, metadatas=metas)
            total_chunks += len(ids)
            print(f"Upserted {total_chunks} chunks so far...")
            ids, docs, metas = [], [], []

    if ids:
        collection.upsert(ids=ids, documents=docs, metadatas=metas)
        total_chunks += len(ids)

    print("Indexing complete.")
    print(f"Collection: {collection_name}")
    print(f"Total chunks indexed: {total_chunks}")
    print(f"Skipped password-protected PDFs: {skipped_protected}")
    print(f"Skipped unreadable PDFs: {skipped_unreadable}")
    print(f"Skipped PDFs with no extractable text: {skipped_no_text}")


def query_context(
    question: str,
    db_dir: str,
    collection_name: str,
    embed_model: str,
    top_k: int,
) -> tuple[str, list[str]]:
    collection = get_collection(db_dir, collection_name, embed_model)
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
    for idx, (doc, meta, distance) in enumerate(
        zip(documents, metadatas, distances), start=1
    ):
        source = str(meta.get("source", "unknown"))
        page = str(meta.get("page", "?"))
        cite = f"{source} (page {page})"
        citations.append(cite)
        context_parts.append(
            f"[{idx}] Source: {cite}\nSimilarity distance: {distance:.4f}\nContent:\n{doc}"
        )

    return "\n\n".join(context_parts), citations


def answer_with_llm(question: str, context: str, model_name: str, api_key: str) -> str:
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is missing. Add it to your .env file.")

    system_prompt = (
        "You are RAGlocal assistant. Use only the provided context to answer. "
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
        model=model_name,
        api_key=api_key,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )

    choice = response["choices"][0]
    message = choice["message"]
    return str(message["content"])


def ask_once(
    question: str,
    db_dir: str,
    collection_name: str,
    embed_model: str,
    llm_model: str,
    api_key: str,
    top_k: int,
) -> None:
    context, citations = query_context(
        question=question,
        db_dir=db_dir,
        collection_name=collection_name,
        embed_model=embed_model,
        top_k=top_k,
    )

    if not context:
        print("No relevant chunks found in vector DB. Run index first or add more PDFs.")
        return

    answer = answer_with_llm(
        question=question,
        context=context,
        model_name=llm_model,
        api_key=api_key,
    )

    print("\nAnswer:\n")
    print(answer)
    print("\nRetrieved citations:\n")
    for cite in dict.fromkeys(citations):
        print(f"- {cite}")


def run_chat(
    db_dir: str,
    collection_name: str,
    embed_model: str,
    llm_model: str,
    api_key: str,
    top_k: int,
) -> None:
    print("RAGlocal chat is ready. Type 'exit' or 'quit' to stop.")
    while True:
        question = input("You: ").strip()
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            print("RAGlocal stopped.")
            break
        ask_once(
            question=question,
            db_dir=db_dir,
            collection_name=collection_name,
            embed_model=embed_model,
            llm_model=llm_model,
            api_key=api_key,
            top_k=top_k,
        )
        print("-" * 80)


def main() -> None:
    load_dotenv()

    pdf_dir_env = os.getenv("RAGLOCAL_PDF_DIR", "")
    db_dir = os.getenv("RAGLOCAL_DB_DIR", "raglocal_db")
    collection_name = os.getenv("RAGLOCAL_COLLECTION", "raglocal_docs")
    embed_model = os.getenv("RAGLOCAL_EMBED_MODEL", "all-MiniLM-L6-v2")
    llm_model = os.getenv("RAGLOCAL_MODEL", "groq/llama-3.3-70b-versatile")
    api_key = os.getenv("GROQ_API_KEY", "")
    top_k = get_env_int("RAGLOCAL_TOP_K", 5)
    chunk_size = get_env_int("RAGLOCAL_CHUNK_SIZE", 1000)
    chunk_overlap = get_env_int("RAGLOCAL_CHUNK_OVERLAP", 150)

    parser = argparse.ArgumentParser(
        description="RAGlocal: local PDF RAG with Chroma vector DB and Groq model via LiteLLM"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    index_cmd = sub.add_parser("index", help="Index PDFs from a directory")
    index_cmd.add_argument(
        "--pdf-dir",
        required=False,
        default=pdf_dir_env,
        help="Directory containing PDF files (defaults to RAGLOCAL_PDF_DIR from .env)",
    )
    index_cmd.add_argument(
        "--reset",
        action="store_true",
        default=True,
        help="Delete existing collection before indexing to avoid mixed old/new documents",
    )
    index_cmd.add_argument(
        "--no-reset",
        dest="reset",
        action="store_false",
        help="Append to existing collection instead of clearing it first",
    )

    ask_cmd = sub.add_parser("ask", help="Ask one question")
    ask_cmd.add_argument("question", help="Question to ask")

    sub.add_parser("chat", help="Start interactive chat mode")

    args = parser.parse_args()

    if args.command == "index":
        if not args.pdf_dir:
            parser.error("Provide --pdf-dir or set RAGLOCAL_PDF_DIR in .env")
        index_pdfs(
            pdf_dir=args.pdf_dir,
            db_dir=db_dir,
            collection_name=collection_name,
            embed_model=embed_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            reset_collection=args.reset,
        )
        return

    if args.command == "ask":
        ask_once(
            question=args.question,
            db_dir=db_dir,
            collection_name=collection_name,
            embed_model=embed_model,
            llm_model=llm_model,
            api_key=api_key,
            top_k=top_k,
        )
        return

    run_chat(
        db_dir=db_dir,
        collection_name=collection_name,
        embed_model=embed_model,
        llm_model=llm_model,
        api_key=api_key,
        top_k=top_k,
    )


if __name__ == "__main__":
    main()
