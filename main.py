
import os
import io
import hashlib

import streamlit as st
import chromadb
import tiktoken

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 RAG Document Assistant")

st.write(
    "Upload documents, process them, and retrieve "
    "relevant information using semantic search."
)


# ==========================================
# 2. CONFIGURATION
# ==========================================

DB_PATH = "./vector_db"
COLLECTION_NAME = "rag_documents"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 80
TOP_K = 4


# ==========================================
# 3. LOAD TOKENIZER
# ==========================================

@st.cache_resource
def load_tokenizer():

    return tiktoken.get_encoding("cl100k_base")


# ==========================================
# 4. LOAD EMBEDDING MODEL
# ==========================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(EMBEDDING_MODEL)


# ==========================================
# 5. CONNECT TO CHROMADB
# ==========================================

@st.cache_resource
def get_collection():

    os.makedirs(DB_PATH, exist_ok=True)

    client = chromadb.PersistentClient(
        path=DB_PATH
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    return collection


# ==========================================
# 6. EXTRACT DOCUMENT TEXT
# ==========================================

def extract_text(uploaded_file):

    filename = uploaded_file.name

    file_bytes = uploaded_file.getvalue()

    pages = []

    try:

        if filename.lower().endswith(".pdf"):

            pdf = PdfReader(
                io.BytesIO(file_bytes)
            )

            for page_number, page in enumerate(
                pdf.pages,
                start=1
            ):

                text = page.extract_text() or ""

                if text.strip():

                    pages.append({
                        "text": text,
                        "page": page_number
                    })

        elif filename.lower().endswith(
            (".txt", ".md")
        ):

            text = file_bytes.decode(
                "utf-8",
                errors="replace"
            )

            if text.strip():

                pages.append({
                    "text": text,
                    "page": 1
                })

        else:

            raise ValueError(
                "Unsupported file format."
            )

        return pages

    except Exception as error:

        st.error(
            f"Error reading {filename}: {error}"
        )

        return []


# ==========================================
# 7. SPLIT TEXT INTO CHUNKS
# ==========================================

def split_text_into_chunks(text):

    tokenizer = load_tokenizer()

    tokens = tokenizer.encode(text)

    chunks = []

    start = 0

    while start < len(tokens):

        end = start + CHUNK_SIZE

        chunk_tokens = tokens[start:end]

        chunk_text = tokenizer.decode(
            chunk_tokens
        )

        if chunk_text.strip():

            chunks.append(
                chunk_text.strip()
            )

        if end >= len(tokens):

            break

        start = end - CHUNK_OVERLAP

    return chunks


# ==========================================
# 8. PROCESS DOCUMENTS
# ==========================================

def process_documents(uploaded_files):

    model = load_embedding_model()

    collection = get_collection()

    total_chunks = 0

    progress = st.progress(0)

    status = st.empty()

    for file_index, uploaded_file in enumerate(
        uploaded_files
    ):

        filename = uploaded_file.name

        status.write(
            f"Processing {filename}..."
        )

        pages = extract_text(uploaded_file)

        if not pages:

            continue

        # Remove old chunks for this filename
        collection.delete(
            where={"source": filename}
        )

        documents = []

        metadatas = []

        ids = []

        for page_data in pages:

            page_text = page_data["text"]

            page_number = page_data["page"]

            chunks = split_text_into_chunks(
                page_text
            )

            for chunk_index, chunk in enumerate(
                chunks
            ):

                unique_value = (
                    f"{filename}_{page_number}_"
                    f"{chunk_index}_{chunk}"
                )

                chunk_id = hashlib.sha256(
                    unique_value.encode("utf-8")
                ).hexdigest()

                documents.append(chunk)

                metadatas.append({
                    "source": filename,
                    "page": int(page_number),
                    "chunk_index": int(chunk_index)
                })

                ids.append(chunk_id)

        if documents:

            embeddings = model.encode(
                documents,
                convert_to_numpy=True
            ).tolist()

            collection.upsert(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )

            total_chunks += len(documents)

        progress.progress(
            (file_index + 1) / len(uploaded_files)
        )

    status.success(
        "Document processing completed!"
    )

    return total_chunks


# ==========================================
# 9. RETRIEVE DOCUMENTS
# ==========================================

def retrieve_documents(question):

    model = load_embedding_model()

    collection = get_collection()

    if collection.count() == 0:

        return []

    query_embedding = model.encode(
        question,
        convert_to_numpy=True
    ).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(
            TOP_K,
            collection.count()
        ),
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    retrieved = []

    documents = results["documents"][0]

    metadatas = results["metadatas"][0]

    distances = results["distances"][0]

    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances
    ):

        retrieved.append({
            "text": document,
            "source": metadata.get(
                "source",
                "Unknown"
            ),
            "page": metadata.get(
                "page",
                "N/A"
            ),
            "distance": distance
        })

    return retrieved


# ==========================================
# 10. SIDEBAR
# ==========================================

with st.sidebar:

    st.header("Project Settings")

    st.write(
        "**Embedding Model**"
    )

    st.code(EMBEDDING_MODEL)

    st.write(
        "**Vector Database**"
    )

    st.code("ChromaDB")

    st.write(
        "**Chunk Size:**",
        CHUNK_SIZE
    )

    st.write(
        "**Chunk Overlap:**",
        CHUNK_OVERLAP
    )

    st.write(
        "**Top K:**",
        TOP_K
    )

    st.divider()

    st.success(
        "Retrieval mode active"
    )

    st.info(
        "No Hugging Face token or OpenAI API key "
        "is required for document retrieval."
    )


# ==========================================
# 11. DOCUMENT UPLOAD
# ==========================================

st.header("1. Upload Documents")

uploaded_files = st.file_uploader(
    "Choose PDF, TXT or MD files",
    type=["pdf", "txt", "md"],
    accept_multiple_files=True
)

if uploaded_files:

    st.subheader("Selected Documents")

    for file in uploaded_files:

        st.write(
            f"📄 {file.name} "
            f"({file.size / 1024:.2f} KB)"
        )

    if st.button(
        "Process Documents",
        type="primary"
    ):

        try:

            with st.spinner(
                "Processing documents..."
            ):

                total_chunks = process_documents(
                    uploaded_files
                )

            st.success(
                f"Successfully processed "
                f"{total_chunks} chunks!"
            )

        except Exception as error:

            st.error(
                f"Processing failed: {error}"
            )


# ==========================================
# 12. DATABASE STATUS
# ==========================================

st.divider()

st.header("2. Knowledge Base Status")

try:

    collection = get_collection()

    count = collection.count()

    col1, col2 = st.columns(2)

    col1.metric(
        "Stored Chunks",
        count
    )

    col2.metric(
        "Top Retrieved Results",
        TOP_K
    )

except Exception as error:

    st.error(
        f"Database error: {error}"
    )


# ==========================================
# 13. ASK QUESTIONS
# ==========================================

st.divider()

st.header("3. Ask Questions")

question = st.text_input(
    "Enter your question",
    placeholder="What is the main topic of my document?"
)

if st.button("Retrieve Relevant Information"):

    if not question.strip():

        st.warning(
            "Please enter a question."
        )

    else:

        try:

            with st.spinner(
                "Searching documents..."
            ):

                results = retrieve_documents(
                    question
                )

            if not results:

                st.warning(
                    "No documents found. "
                    "Please upload and process documents first."
                )

            else:

                st.success(
                    f"Retrieved {len(results)} relevant chunks."
                )

                st.subheader(
                    "Retrieved Information"
                )

                for index, result in enumerate(
                    results,
                    start=1
                ):

                    with st.expander(
                        f"Result {index}: {result['source']}"
                    ):

                        st.write(
                            result["text"]
                        )

                        st.caption(
                            f"Source: {result['source']} | "
                            f"Page: {result['page']} | "
                            f"Distance: {result['distance']:.4f}"
                        )

                st.divider()

                st.subheader("Sources")

                unique_sources = {}

                for result in results:

                    source = result["source"]

                    if source not in unique_sources:

                        unique_sources[source] = set()

                    unique_sources[source].add(
                        str(result["page"])
                    )

                for source, pages in unique_sources.items():

                    page_list = ", ".join(
                        sorted(pages)
                    )

                    st.write(
                        f"📄 **{source}** — Pages: {page_list}"
                    )

                st.info(
                    "Document retrieval completed. "
                    "AI-generated answer generation is disabled "
                    "in this deployment version."
                )

        except Exception as error:

            st.error(
                f"Retrieval failed: {error}"
            )


# ==========================================
# 14. FOOTER
# ==========================================

st.divider()

st.caption(
    "RAG Document Assistant | "
    "Python | Streamlit | ChromaDB | "
    "Sentence Transformers"
)
