
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
    "Upload documents, ask questions, and retrieve "
    "relevant answers from your documents."
)


# ==========================================
# 2. CONFIGURATION
# ==========================================

CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
TOP_K = 4
MAX_FILE_SIZE_MB = 10


# ==========================================
# 3. SESSION STATE
# ==========================================

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

if "last_answer" not in st.session_state:
    st.session_state.last_answer = None


# ==========================================
# 4. TEXT TOKENIZATION
# ==========================================

def tokenize(text):

    return re.findall(
        r"\b[a-zA-Z0-9]+\b",
        text.lower()
    )


# ==========================================
# 5. TEXT EXTRACTION
# ==========================================

def extract_text(uploaded_file):

    filename = uploaded_file.name

    file_extension = filename.lower().split(".")[-1]

    if file_extension == "pdf":

        pdf_reader = PdfReader(
            io.BytesIO(uploaded_file.getvalue())
        )

        extracted_pages = []

        for page_number, page in enumerate(
            pdf_reader.pages,
            start=1
        ):

            text = page.extract_text() or ""

            if text.strip():

                extracted_pages.append({
                    "text": text,
                    "page": page_number
                })

        return extracted_pages

    elif file_extension in ["txt", "md"]:

        content = uploaded_file.getvalue().decode(
            "utf-8",
            errors="replace"
        )

        return [{
            "text": content,
            "page": 1
        }]

    else:

        raise ValueError(
            "Unsupported file format. "
            "Please upload PDF, TXT, or MD files."
        )


# ==========================================
# 6. DOCUMENT CHUNKING
# ==========================================

def create_chunks(text, source, page):

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = start + CHUNK_SIZE

        chunk_text = " ".join(
            words[start:end]
        )

        if chunk_text.strip():

            chunks.append({
                "text": chunk_text,
                "source": source,
                "page": page
            })

        if end >= len(words):
            break

        start = end - CHUNK_OVERLAP

    return chunks


# ==========================================
# 7. DOCUMENT PROCESSING
# ==========================================

def process_documents(uploaded_files):

    all_chunks = []

    processed_files = []

    for uploaded_file in uploaded_files:

        if uploaded_file.size > MAX_FILE_SIZE_MB * 1024 * 1024:

            raise ValueError(
                f"{uploaded_file.name} exceeds "
                f"the {MAX_FILE_SIZE_MB} MB limit."
            )

        pages = extract_text(uploaded_file)

        if not pages:

            continue

        file_chunks = []

        for page_data in pages:

            page_chunks = create_chunks(
                page_data["text"],
                uploaded_file.name,
                page_data["page"]
            )

            file_chunks.extend(page_chunks)

        all_chunks.extend(file_chunks)

        if file_chunks:

            processed_files.append(
                uploaded_file.name
            )

    return all_chunks, processed_files


# ==========================================
# 8. BM25 DOCUMENT RETRIEVAL
# ==========================================


def retrieve_documents(question):

    documents = st.session_state.chunks

    if not documents:
        return []

    def normalize_word(word):
        word = word.lower()

        if len(word) > 4 and word.endswith("ies"):
            return word[:-3] + "y"

        if len(word) > 3 and word.endswith("s"):
            return word[:-1]

        return word

    query_terms = [
        normalize_word(word)
        for word in tokenize(question)
    ]

    query_terms = list(set(query_terms))

    if not query_terms:
        return []

    tokenized_docs = [
        [normalize_word(word) for word in tokenize(item["text"])]
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

        for term in query_terms:

            frequency = term_counts.get(term, 0)

            if frequency == 0:
                continue

            df = document_frequencies[term]

            idf = math.log(
                1 + (document_count - df + 0.5) / (df + 0.5)
            )

            denominator = frequency + k1 * (
                1 - b + b * (document_length / average_length)
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
# 9. QUESTION-FOCUSED ANSWER EXTRACTION
# ==========================================


def generate_document_answer(question, results):

    if not results:
        return {
            "answer": (
                "No matching document sections were found. "
                "Try using important terms from your document."
            ),
            "sources": [],
            "evidence": []
        }

    stop_words = {
        "what", "is", "are", "was", "were",
        "the", "a", "an", "of", "in", "on",
        "at", "to", "for", "and", "or",
        "how", "why", "when", "where", "who",
        "which", "explain", "describe", "tell",
        "me", "about", "does", "do", "did",
        "can", "could", "would", "should",
        "has", "have", "had", "with", "from",
        "this", "that", "these", "those",
        "give", "list", "define", "discuss",
        "write", "mention", "briefly"
    }

    def normalize_word(word):

        word = word.lower()

        if len(word) > 4 and word.endswith("ies"):
            return word[:-3] + "y"

        if len(word) > 3 and word.endswith("s"):
            return word[:-1]

        return word

    query_terms = {
        normalize_word(word)
        for word in tokenize(question)
        if word not in stop_words
    }

    if not query_terms:
        return {
            "answer": "Please enter a more specific question.",
            "sources": [],
            "evidence": []
        }

    candidates = []

    for result in results:

        sentences = re.split(
            r"(?<=[.!?])\s+",
            result["text"]
        )

        for sentence in sentences:

            sentence = sentence.strip()

            if len(sentence.split()) < 4:
                continue

            sentence_tokens = {
                normalize_word(word)
                for word in tokenize(sentence)
                if word not in stop_words
            }

            if not sentence_tokens:
                continue

            matching_terms = query_terms.intersection(
                sentence_tokens
            )

            if not matching_terms:
                continue

            coverage = (
                len(matching_terms) / len(query_terms)
            )

            relevance = (
                len(matching_terms) / len(sentence_tokens)
            )

            score = (
                coverage * 0.75
                + relevance * 0.25
            )

            candidates.append({
                "text": sentence,
                "score": score,
                "source": result["source"],
                "page": result["page"]
            })

    candidates.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    if not candidates:
        return {
            "answer": (
                "No exact keyword matches were found. "
                "Try asking with important words from the document."
            ),
            "sources": [],
            "evidence": []
        }

    selected_sentences = []
    evidence = []
    sources = []
    seen_sentences = set()

    for candidate in candidates:

        sentence = candidate["text"]

        if sentence.lower() in seen_sentences:
            continue

        seen_sentences.add(sentence.lower())

        selected_sentences.append(sentence)
        evidence.append(candidate)

        source_info = {
            "source": candidate["source"],
            "page": candidate["page"]
        }

        if source_info not in sources:
            sources.append(source_info)

        if len(selected_sentences) >= 3:
            break

    if candidates[0]["score"] < 0.15:

        answer = (
            "I found some potentially related information, "
            "but it may not fully answer your question:\n\n"
        ) + " ".join(selected_sentences)

    else:

        answer = " ".join(selected_sentences)

    return {
        "answer": answer,
        "sources": sources,
        "evidence": evidence
    }


# ==========================================
# 10. SIDEBAR
# ==========================================

with st.sidebar:

    st.header("About Project")

    st.write(
        "RAG Document Assistant uses BM25 keyword "
        "retrieval and question-focused sentence extraction."
    )

    st.divider()

    st.subheader("Technologies")

    st.write("• Python")
    st.write("• Streamlit")
    st.write("• PyPDF")
    st.write("• BM25 Retrieval")

    st.divider()

    st.write(
        f"Processed documents: "
        f"{len(st.session_state.processed_files)}"
    )

    st.write(
        f"Total chunks: "
        f"{len(st.session_state.chunks)}"
    )

    if st.button("Clear Documents"):

        st.session_state.chunks = []

        st.session_state.processed_files = []

        st.session_state.last_answer = None

        st.rerun()


# ==========================================
# 11. DOCUMENT UPLOAD
# ==========================================

st.header("1. Upload Documents")

uploaded_files = st.file_uploader(
    "Upload your PDF, TXT, or MD documents",
    type=["pdf", "txt", "md"],
    accept_multiple_files=True
)


# ==========================================
# 12. PROCESS DOCUMENTS
# ==========================================

st.header("2. Process Documents")

if st.button(
    "Process Documents",
    type="primary"
):

    if not uploaded_files:

        st.warning(
            "Please upload at least one document."
        )

    else:

        try:

            with st.spinner(
                "Extracting and processing documents..."
            ):

                chunks, processed_files = (
                    process_documents(uploaded_files)
                )

            if not chunks:

                st.warning(
                    "No readable text found in the uploaded documents."
                )

            else:

                st.session_state.chunks = chunks

                st.session_state.processed_files = (
                    processed_files
                )

                st.session_state.last_answer = None

                st.success(
                    "Documents processed successfully!"
                )

                st.write(
                    f"Total chunks created: {len(chunks)}"
                )

                st.write("Processed files:")

                for filename in processed_files:

                    st.write(f"📄 {filename}")

        except Exception as error:

            st.error(
                f"Document processing failed: {error}"
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

if st.button("Get Answer"):

    if not question.strip():

        st.warning(
            "Please enter a question."
        )

    elif not st.session_state.chunks:

        st.warning(
            "Please upload and process documents first."
        )

    else:

        try:

            with st.spinner(
                "Searching documents and extracting an answer..."
            ):

                results = retrieve_documents(question)

                answer_result = generate_document_answer(
                    question,
                    results
                )

                st.session_state.last_answer = (
                    answer_result
                )

            st.success(
                "Question processing completed!"
            )

        except Exception as error:

            st.error(
                f"Answer extraction failed: {error}"
            )


# ==========================================
# 14. DISPLAY ANSWER
# ==========================================

if st.session_state.last_answer:

    answer_result = st.session_state.last_answer

    st.subheader("Answer")

    st.info(
        answer_result["answer"]
    )

    if answer_result["sources"]:

        st.subheader("Answer Sources")

        for source in answer_result["sources"]:

            st.write(
                f"📄 **{source['source']}** "
                f"— Page: {source['page']}"
            )

    if answer_result["evidence"]:

        with st.expander(
            "View supporting document sentences"
        ):

            for index, item in enumerate(
                answer_result["evidence"],
                start=1
            ):

                st.write(
                    f"**{index}.** {item['text']}"
                )

                st.caption(
                    f"Source: {item['source']} | "
                    f"Page: {item['page']}"
                )


# ==========================================
# 15. FOOTER
# ==========================================

st.divider()

st.caption(
    "RAG Document Assistant | "
    "Python | Streamlit | PyPDF | BM25 | "
    "Question-Focused Answer Extraction"
)
