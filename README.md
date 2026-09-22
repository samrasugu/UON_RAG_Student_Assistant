# UoN Student Assistant — Simple RAG Demo

A small classroom RAG system built with Python, OpenAI embeddings, Chroma and an OpenAI Responses API model.

## 1. Create an environment

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

## 3. Configure credentials

Copy `.env.example` to `.env` and set:

```text
OPENAI_API_KEY=...
OPENAI_CHAT_MODEL=...
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

Use a model available to your API account for `OPENAI_CHAT_MODEL`.

## 4. Run

```bash
python rag_demo.py
```

The script:
1. Reads `.txt` documents from `knowledge_base/`
2. Splits them into chunks
3. Creates embeddings
4. Stores them in a local Chroma database
5. Embeds the user's question
6. Retrieves the most similar chunks
7. Sends the question + retrieved context to the LLM
8. Prints the answer and source filenames

## 5. Classroom questions

Try:
- What are the requirements for the final-year project?
- When is the project proposal due?
- What should a student include in the project report?
- What is the library opening time?
- What is the refund policy?  <-- deliberately outside the knowledge base

The last question demonstrates that RAG can still fail when the source corpus does not contain the answer.

## Important teaching note

This is a teaching demo, not a production architecture. In production you would add authentication, authorization, better document parsing, metadata filters, evaluation, monitoring, retries, rate-limit handling, secrets management and protection against prompt injection/data poisoning.
