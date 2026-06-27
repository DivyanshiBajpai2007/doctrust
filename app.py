import streamlit as st
from PyPDF2 import PdfReader
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings
import re
import os

# ── Page config ──
st.set_page_config(
    page_title="DocuTrust",
    page_icon="📄",
    layout="wide"
)

# ── API Key ──
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

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

# ── Session state ──
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False

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

            embeddings = FastEmbedEmbeddings()
            vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=embeddings,
                persist_directory="./chroma_db"
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
    st.markdown("### How it works")
    st.markdown("""
    1. Upload PDF
    2. Text extracted and split into chunks
    3. Chunks stored as vectors in ChromaDB
    4. Your question converted to vector
    5. Cosine similarity finds closest chunks
    6. Relevance checked (CRAG pattern)
    7. AI answers from document only
    8. Source shown with answer
    """)

    st.divider()
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ── Main area ──
st.title("DocuTrust")
st.caption("Enterprise Advanced RAG Platform with Self-Correction | Ask questions from your documents")
st.divider()

if not st.session_state.pdf_processed:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**Step 1**\nUpload a PDF from the sidebar")
    with col2:
        st.success("**Step 2**\nAI reads and indexes the document")
    with col3:
        st.warning("**Step 3**\nAsk any question about the document")

    st.markdown("### What can DocuTrust do?")
    for f in [
        "Answer questions from any PDF document",
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

    # Suggested questions
    if len(st.session_state.messages) == 0:
        st.markdown("### Try asking:")
        col1, col2 = st.columns(2)
        suggestions = [
            "What is this document about?",
            "Summarize the main points",
            "What are the key findings?",
            "What are the conclusions?",
        ]
        for i, s in enumerate(suggestions):
            if i % 2 == 0:
                if col1.button(s, use_container_width=True, key=s):
                    st.session_state.messages.append({"role": "user", "content": s})
                    st.rerun()
            else:
                if col2.button(s, use_container_width=True, key=s):
                    st.session_state.messages.append({"role": "user", "content": s})
                    st.rerun()

    # Chat input
    if prompt := st.chat_input("Ask anything about your document..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching document..."):

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
                    model="llama-3.3-70b-versatile",
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
                    answer = response.content
                    st.markdown(answer)

                    with st.expander("Source chunks from document"):
                        for i, doc in enumerate(retrieved_docs):
                            st.markdown(f"**Chunk {i+1}:**")
                            st.markdown(doc.page_content[:300] + "...")
                            st.divider()

                else:
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
                    answer = response2.content
                    st.markdown(answer)
                    st.caption(f"Query rewritten for better results: {rewritten_query}")

                    with st.expander("Source chunks from document"):
                        for i, doc in enumerate(retrieved_docs2):
                            st.markdown(f"**Chunk {i+1}:**")
                            st.markdown(doc.page_content[:300] + "...")
                            st.divider()

        st.session_state.messages.append({"role": "assistant", "content": answer})