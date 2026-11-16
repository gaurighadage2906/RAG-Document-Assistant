
import io
import re
import math
import streamlit as st
from collections import Counter
from pypdf import PdfReader


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
    "Upload PDF, TXT or Markdown documents, "
    "process them and retrieve relevant passages."
)

CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
TOP_K = 4
MAX_FILE_SIZE = 10 * 1024 * 1024


# ==========================================
# 2. SESSION STORAGE
# ==========================================

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "processed_files" not in st.session_state:
    st.session_state.processed_files = []


# ==========================================
# 3. TEXT EXTRACTION
# ==========================================

def extract_text(uploaded_file):

    filename = uploaded_file.name
    file_bytes = uploaded_file.getvalue()

    if filename.lower().endswith(".pdf"):

        reader = PdfReader(io.BytesIO(file_bytes))

        pages = []

        for page_number, page in enumerate(
            reader.pages, start=1
        ):

            text = page.extract_text() or ""

            if text.strip():
                pages.append({
                    "text": text,
                    "page": page_number
                })

        return pages

    if filename.lower().endswith((".txt", ".md")):

        text = file_bytes.decode(
            "utf-8",
            errors="replace"
        )

        return [{
            "text": text,
            "page": 1
        }] if text.strip() else []

    raise ValueError("Unsupported file format.")


# ==========================================
# 4. TEXT CHUNKING
# ==========================================

def split_into_chunks(text):

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = start + CHUNK_SIZE

        chunk = " ".join(words[start:end])

        if chunk.strip():
            chunks.append(chunk)

        if end >= len(words):
            break

        start = end - CHUNK_OVERLAP

    return chunks


# ==========================================
# 5. PROCESS DOCUMENTS
# ==========================================

def process_documents(uploaded_files):

    all_chunks = []

    progress = st.progress(0)

    status = st.empty()

    for index, uploaded_file in enumerate(uploaded_files):

        filename = uploaded_file.name

        status.write(f"Processing {filename}...")

        if uploaded_file.size > MAX_FILE_SIZE:
            st.warning(
                f"Skipped {filename}: maximum size is 10 MB."
            )
            progress.progress(
                (index + 1) / len(uploaded_files)
            )
            continue

        try:

            pages = extract_text(uploaded_file)

            for page_data in pages:

                page_chunks = split_into_chunks(
                    page_data["text"]
                )

                for chunk_index, chunk in enumerate(
                    page_chunks
                ):

                    all_chunks.append({
                        "text": chunk,
                        "source": filename,
                        "page": page_data["page"],
                        "chunk_index": chunk_index
                    })

        except Exception as error:

            st.warning(
                f"Could not process {filename}: {error}"
            )

        progress.progress(
            (index + 1) / len(uploaded_files)
        )

    # Replace the current knowledge base
    st.session_state.chunks = all_chunks

    st.session_state.processed_files = [
        file.name for file in uploaded_files
    ]

    status.success("Document processing completed!")

    return len(all_chunks)


# ==========================================
# 6. TEXT TOKENIZATION
# ==========================================

def tokenize(text):

    return re.findall(
        r"[a-zA-Z0-9]+",
        text.lower()
    )


# ==========================================
# 7. BM25 RETRIEVAL
# ==========================================

def retrieve_documents(question):

    documents = st.session_state.chunks

    if not documents:
        return []

    query_terms = tokenize(question)

    if not query_terms:
        return []

    tokenized_docs = [
        tokenize(item["text"])
        for item in documents
    ]

    document_count = len(tokenized_docs)

    average_length = (
        sum(len(doc) for doc in tokenized_docs)
        / document_count
    )

    if average_length == 0:
        return []

    document_frequencies = Counter()

    for tokens in tokenized_docs:
        document_frequencies.update(set(tokens))

    k1 = 1.5
    b = 0.75

    scored_results = []

    for index, tokens in enumerate(tokenized_docs):

        term_counts = Counter(tokens)

        document_length = len(tokens)

        score = 0.0

        for term in set(query_terms):

            frequency = term_counts.get(term, 0)

            if frequency == 0:
                continue

            df = document_frequencies[term]

            idf = math.log(
                1 + (
                    document_count - df + 0.5
                ) / (df + 0.5)
            )

            denominator = frequency + k1 * (
                1 - b + b * (
                    document_length / average_length
                )
            )

            score += idf * (
                frequency * (k1 + 1)
            ) / denominator

        if score > 0:

            result = documents[index].copy()

            result["score"] = score

            scored_results.append(result)

    scored_results.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    return scored_results[:TOP_K]


# ==========================================
# 8. SIDEBAR
# ==========================================

with st.sidebar:

    st.header("Project Settings")

    st.write("**Retrieval Method**")
    st.code("BM25 Keyword Retrieval")

    st.write("**Chunk Size:**", CHUNK_SIZE)

    st.write("**Chunk Overlap:**", CHUNK_OVERLAP)

    st.write("**Top Results:**", TOP_K)

    st.divider()

    st.success("Lightweight retrieval mode active")

    st.info(
        "No OpenAI key, Hugging Face token, "
        "or embedding model is required."
    )


# ==========================================
# 9. DOCUMENT UPLOAD
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

    if st.button("Process Documents", type="primary"):

        try:

            with st.spinner("Processing documents..."):

                total_chunks = process_documents(
                    uploaded_files
                )

            st.success(
                f"Successfully processed {total_chunks} chunks!"
            )

        except Exception as error:

            st.error(
                f"Processing failed: {error}"
            )


# ==========================================
# 10. KNOWLEDGE BASE STATUS
# ==========================================

st.divider()

st.header("2. Knowledge Base Status")

col1, col2 = st.columns(2)

col1.metric(
    "Stored Chunks",
    len(st.session_state.chunks)
)

col2.metric(
    "Processed Documents",
    len(st.session_state.processed_files)
)

if st.session_state.processed_files:

    st.write("**Processed files:**")

    for filename in st.session_state.processed_files:
        st.write(f"📄 {filename}")


# ==========================================
# 11. ASK QUESTIONS
# ==========================================

st.divider()

st.header("3. Ask Questions")

question = st.text_input(
    "Enter your question",
    placeholder="What is the main topic of my document?"
)

if st.button("Retrieve Relevant Information"):

    if not question.strip():

        st.warning("Please enter a question.")

    elif not st.session_state.chunks:

        st.warning(
            "Please upload and process documents first."
        )

    else:

        try:

            with st.spinner("Searching documents..."):

                results = retrieve_documents(question)

            if not results:

                st.warning(
                    "No matching passages found. "
                    "Try using different keywords."
                )

            else:

                st.success(
                    f"Retrieved {len(results)} relevant passages."
                )

                st.subheader("Retrieved Information")

                for index, result in enumerate(
                    results, start=1
                ):

                    with st.expander(
                        f"Result {index}: {result['source']}"
                    ):

                        st.write(result["text"])

                        st.caption(
                            f"Source: {result['source']} | "
                            f"Page: {result['page']} | "
                            f"BM25 Score: {result['score']:.4f}"
                        )

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
                        sorted(pages, key=lambda x: int(x))
                    )

                    st.write(
                        f"📄 **{source}** — Pages: {page_list}"
                    )

                st.info(
                    "Retrieval completed. "
                    "This lightweight deployment uses keyword-based "
                    "BM25 retrieval, not semantic embeddings or "
                    "AI-generated answers."
                )
        except Exception as error:
            st.error(f"Retrieval failed: {error}")


# ==========================================
# 12. FOOTER
# ==========================================

st.divider()

st.caption(
    "RAG Document Assistant | "
    "Python | Streamlit | PyPDF | BM25"
)
