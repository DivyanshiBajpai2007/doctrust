import streamlit as st
from PyPDF2 import PdfReader
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings
from groq import Groq
from gtts import gTTS
import io
import re
import os
import uuid

# ── Page config ──
st.set_page_config(
    page_title="DocuTrust",
    page_icon="📄",
    layout="wide"
)

# ── API Key ──
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
# 💡 Remember: Groq retires models over time, so keep the name in one place
LLM_MODEL = "openai/gpt-oss-120b"

# ── Manual text splitter (no langchain_text_splitters needed) ──
# 💡 Remember: We split because AI has a token limit
# chunk_size = how many characters per chunk
# overlap = shared characters between chunks so context isn't lost
def split_text(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    return chunks

# ── Voice: speech-to-text ──
# 💡 Remember: Whisper turns the recorded audio into text,
# then the question goes through the exact same RAG pipeline as a typed one
def transcribe(audio_bytes):
    client = Groq(api_key=GROQ_API_KEY)
    result = client.audio.transcriptions.create(
        file=("question.wav", audio_bytes),
        model="whisper-large-v3-turbo",
    )
    return result.text.strip()

# ── Voice: text-to-speech ──
# 💡 Remember: strip markdown first so the voice doesn't read out "*" and "#",
# and switch to Hindi if the answer is written in Devanagari
def speak(text):
    clean = re.sub(r"[*#`_>\[\]]", "", text)
    lang = "hi" if re.search(r"[ऀ-ॿ]", clean) else "en"
    buf = io.BytesIO()
    gTTS(clean, lang=lang).write_to_fp(buf)
    return buf.getvalue()

# ── RAG pipeline (shared by typed, spoken and suggested questions) ──
def answer_question(prompt):
    # Step 4: Search ChromaDB
    # 💡 Remember: This is cosine similarity in action!
    # Question gets embedded → ChromaDB finds closest chunk vectors
    retriever = st.session_state.vectorstore.as_retriever(
        search_kwargs={"k": 4}
    )
    retrieved_docs = retriever.invoke(prompt)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    # Initialize LLM
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model=LLM_MODEL,
        temperature=0
    )

    # Step 5: Grade relevance (CRAG pattern)
    # 💡 Remember: This is the corrective part of CRAG
    # Checking if retrieved vectors are actually relevant before answering
    grade_prompt = f"""You are a relevance grader.
    Question: {prompt}
    Retrieved chunks: {context[:500]}
    Are these chunks relevant to answer the question? Reply only 'yes' or 'no'."""

    grade_response = llm.invoke(grade_prompt)
    is_relevant = "yes" in grade_response.content.lower()

    if is_relevant:
        # Step 6: Generate answer from document
        # 💡 Remember: Grounded generation — answers only from YOUR document
        answer_prompt = f"""You are DocuTrust, an expert document analysis assistant.
Answer the question based ONLY on the provided document context.
If the answer is in the context, provide it clearly.
If you cannot find the answer, say so honestly.

Context from document:
{context}

Question: {prompt}

Provide a clear answer and mention which part of the document supports it."""

        response = llm.invoke(answer_prompt)
        return response.content, retrieved_docs, None

    # Step 7: Query rewriting if not relevant
    # 💡 Remember: CRAG pattern — rewrite and retry if retrieval fails
    rewrite_prompt = f"""Rewrite this question to be more specific for document search:
Original: {prompt}
Rewritten:"""
    rewrite_response = llm.invoke(rewrite_prompt)
    rewritten_query = rewrite_response.content

    retrieved_docs2 = retriever.invoke(rewritten_query)
    context2 = "\n\n".join([doc.page_content for doc in retrieved_docs2])

    answer_prompt2 = f"""You are DocuTrust, an expert document analysis assistant.
Answer based on document context. If not found, say so clearly.

Context: {context2}
Question: {prompt}
Answer:"""

    response2 = llm.invoke(answer_prompt2)
    return response2.content, retrieved_docs2, rewritten_query

def show_sources(docs):
    with st.expander("Source chunks from document"):
        for i, doc in enumerate(docs):
            st.markdown(f"**Chunk {i+1}:**")
            st.markdown(doc.page_content[:300] + "...")
            st.divider()

# ── Session state ──
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False
if "pending" not in st.session_state:
    st.session_state.pending = None
# 💡 Remember: changing the mic widget's key resets it, so the same recording
# isn't transcribed again on every rerun
if "audio_key" not in st.session_state:
    st.session_state.audio_key = 0

# ── Sidebar ──
with st.sidebar:
    st.markdown("## DocuTrust")
    st.markdown("**Enterprise RAG Platform**")
    st.markdown("Built by **Divyanshi Bajpai**")
    st.markdown("3rd Year CSE-AI | MPGI Kanpur")
    st.divider()

    st.markdown("### Upload Document")
    uploaded_file = st.file_uploader(
        "Upload a PDF file",
        type=["pdf"],
        help="Upload any PDF document to start asking questions"
    )

    if uploaded_file and not st.session_state.pdf_processed:
        with st.spinner("Processing PDF..."):

            # Step 1: Read PDF
            # 💡 Remember: PyPDF2 extracts raw text from each page
            pdf_reader = PdfReader(uploaded_file)
            raw_text = ""
            for page in pdf_reader.pages:
                text = page.extract_text()
                if text:
                    raw_text += text

            st.info(f"Extracted {len(raw_text)} characters from PDF")

            # Step 2: Split into chunks
            # 💡 Remember: This is like cutting a book into paragraphs
            # So the AI can search through smaller pieces efficiently
            chunks = split_text(raw_text, chunk_size=1000, overlap=200)
            st.info(f"Split into {len(chunks)} chunks")

            # Step 3: Store in ChromaDB using FastEmbed
            # 💡 Remember: FastEmbed converts text to vectors using ONNX
            # ChromaDB stores these vectors and uses cosine similarity to find matches
            from langchain_core.documents import Document
            documents = [Document(page_content=chunk) for chunk in chunks]

            # 💡 Remember: each upload gets its own in-memory collection, so chunks
            # from an earlier PDF (or another user's PDF) never leak into answers
            embeddings = FastEmbedEmbeddings()
            vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=embeddings,
                collection_name=f"doc_{uuid.uuid4().hex}"
            )
            st.session_state.vectorstore = vectorstore
            st.session_state.pdf_processed = True

        st.success("PDF processed and stored in ChromaDB!")
        st.markdown(f"**File:** {uploaded_file.name}")
        st.markdown(f"**Chunks:** {len(chunks)}")

    if st.session_state.pdf_processed:
        st.success("Document ready!")
        if st.button("Clear and Upload New PDF", use_container_width=True):
            st.session_state.pdf_processed = False
            st.session_state.vectorstore = None
            st.session_state.messages = []
            st.rerun()

    st.divider()
    st.markdown("### Voice")
    speak_answers = st.toggle("🔊 Speak answers", value=True,
                              help="Read each answer aloud (gTTS)")

    st.divider()
    st.markdown("### How it works")
    st.markdown("""
    1. Upload PDF
    2. Text extracted and split into chunks
    3. Chunks stored as vectors in ChromaDB
    4. Type or speak your question (Whisper speech-to-text)
    5. Cosine similarity finds closest chunks
    6. Relevance checked (CRAG pattern)
    7. AI answers from document only
    8. Source shown with answer, read aloud if voice is on
    """)

    st.divider()
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ── Main area ──
st.title("DocuTrust")
st.caption("Enterprise Advanced RAG Platform with Self-Correction | Type or speak questions about your documents")
st.divider()

if not st.session_state.pdf_processed:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**Step 1**\nUpload a PDF from the sidebar")
    with col2:
        st.success("**Step 2**\nAI reads and indexes the document")
    with col3:
        st.warning("**Step 3**\nType or speak any question about the document")

    st.markdown("### What can DocuTrust do?")
    for f in [
        "Answer questions from any PDF document",
        "Ask by voice and hear the answer read aloud",
        "Show exactly which part of the document the answer came from",
        "Check if retrieved information is relevant (CRAG pattern)",
        "Rewrite queries if initial retrieval fails",
        "Maintain conversation history for follow-up questions",
    ]:
        st.markdown(f"- {f}")

else:
    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("audio"):
                st.audio(msg["audio"], format="audio/mp3")

    # New turns render here, above the mic widget
    chat_area = st.container()

    # Suggested questions
    if len(st.session_state.messages) == 0 and not st.session_state.pending:
        st.markdown("### Try asking:")
        col1, col2 = st.columns(2)
        suggestions = [
            "What is this document about?",
            "Summarize the main points",
            "What are the key findings?",
            "What are the conclusions?",
        ]
        for i, s in enumerate(suggestions):
            col = col1 if i % 2 == 0 else col2
            if col.button(s, use_container_width=True, key=s):
                st.session_state.pending = s
                st.rerun()

    # Voice input
    audio = st.audio_input("🎤 Or ask by voice", key=f"mic_{st.session_state.audio_key}")
    if audio is not None:
        with st.spinner("Transcribing..."):
            try:
                spoken = transcribe(audio.getvalue())
            except Exception as e:
                spoken = ""
                st.error(f"Could not transcribe audio: {e}")
        st.session_state.audio_key += 1
        if spoken:
            st.session_state.pending = "🎤 " + spoken
            st.rerun()
        else:
            st.warning("Didn't catch that. Please try recording again.")

    # Chat input
    if prompt := st.chat_input("Ask anything about your document..."):
        st.session_state.pending = prompt

    # Answer whichever question came in (typed, spoken or suggested)
    if st.session_state.pending:
        shown = st.session_state.pending
        prompt = shown.removeprefix("🎤 ")
        st.session_state.pending = None
        st.session_state.messages.append({"role": "user", "content": shown})

        with chat_area:
            with st.chat_message("user"):
                st.markdown(shown)

            with st.chat_message("assistant"):
                with st.spinner("Searching document..."):
                    answer, docs, rewritten_query = answer_question(prompt)
                st.markdown(answer)
                if rewritten_query:
                    st.caption(f"Query rewritten for better results: {rewritten_query}")
                show_sources(docs)

                audio_reply = None
                if speak_answers:
                    try:
                        audio_reply = speak(answer)
                        st.audio(audio_reply, format="audio/mp3", autoplay=True)
                    except Exception as e:
                        st.caption(f"Voice reply unavailable: {e}")

        st.session_state.messages.append({"role": "assistant", "content": answer, "audio": audio_reply})
