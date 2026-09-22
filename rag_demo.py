import os
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL")
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

if not API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY in .env")
if not CHAT_MODEL:
    raise RuntimeError("Set OPENAI_CHAT_MODEL in .env")

client = OpenAI(api_key=API_KEY)
chroma = chromadb.PersistentClient(path="./chroma_db")
collection = chroma.get_or_create_collection(name="uon_student_assistant")

KB_DIR = Path("knowledge_base")


def read_documents():
    docs = []
    for path in KB_DIR.glob("*.txt"):
        docs.append((path.name, path.read_text(encoding="utf-8")))
    if not docs:
        raise RuntimeError("No .txt files found in knowledge_base/")
    return docs


def chunk_text(text, chunk_size=700, overlap=120):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = end - overlap
    return chunks


def embed(texts):
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )
    return [item.embedding for item in response.data]


def build_index():
    docs = read_documents()
    # Rebuild the classroom index each time.
    try:
        chroma.delete_collection("uon_student_assistant")
    except Exception:
        pass
    collection = chroma.get_or_create_collection(name="uon_student_assistant")

    all_chunks = []
    metadatas = []
    ids = []

    for filename, text in docs:
        for i, chunk in enumerate(chunk_text(text)):
            all_chunks.append(chunk)
            metadatas.append({"source": filename, "chunk": i})
            ids.append(f"{filename}-{i}")

    vectors = embed(all_chunks)

    collection.add(
        ids=ids,
        documents=all_chunks,
        embeddings=vectors,
        metadatas=metadatas,
    )
    return collection


def retrieve(collection, question, k=4):
    q_vector = embed([question])[0]
    result = collection.query(
        query_embeddings=[q_vector],
        n_results=k,
    )
    docs = result["documents"][0]
    metas = result["metadatas"][0]
    distances = result["distances"][0]

    return list(zip(docs, metas, distances))


def answer_with_rag(question, retrieved):
    context_parts = []
    for i, (doc, meta, distance) in enumerate(retrieved, 1):
        context_parts.append(
            f"[SOURCE {i}: {meta['source']} | chunk {meta['chunk']}]\n{doc}"
        )
    context = "\n\n".join(context_parts)

    prompt = f"""You are a university student assistant.

Answer the user's question using ONLY the supplied context.
If the context does not contain enough information, say:
"I don't have enough information in the provided documents."

Do not invent requirements, dates, rules, names or policies.
At the end, list the source filenames you used.

CONTEXT:
{context}

QUESTION:
{question}
"""

    response = client.responses.create(
        model=CHAT_MODEL,
        input=prompt,
    )
    return response.output_text


def main():
    print("\n=== UoN Student Assistant — Simple RAG Demo ===\n")

    collection = build_index()
    print("Knowledge base indexed.\n")

    question = input("Ask a question: ").strip()
    if not question:
        return

    retrieved = retrieve(collection, question, k=4)

    print("\n--- Retrieved chunks ---\n")
    for i, (doc, meta, distance) in enumerate(retrieved, 1):
        print(f"[{i}] {meta['source']} | chunk {meta['chunk']} | distance={distance:.4f}")
        print(doc[:500] + ("..." if len(doc) > 500 else ""))
        print()

    answer = answer_with_rag(question, retrieved)

    print("\n--- Grounded answer ---\n")
    print(answer)


if __name__ == "__main__":
    main()
