import streamlit as st
from pathlib import Path
import json
import uuid
from core.extraction import extract_text_pages
from core.entities import extract_entities
from core.risks import analyze_risks
from core.summarization import summarize_document
from core.pdf_writer import save_json_to_pdf
from rag.chunking import chunk_pages
from rag.store import upsert_chunks
from rag.qa import answer

st.set_page_config(page_title="ClauseIQ", layout="wide")

# -----------------------
# Session state setup
# -----------------------
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False
if "json_result" not in st.session_state:
    st.session_state.json_result = None
if "pdf_path" not in st.session_state:
    st.session_state.pdf_path = None
if "doc_id" not in st.session_state:
    st.session_state.doc_id = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "filename" not in st.session_state:
    st.session_state.filename = None


def reset_app():
    st.session_state.analysis_done = False
    st.session_state.json_result = None
    st.session_state.pdf_path = None
    st.session_state.doc_id = None
    st.session_state.chat_history = []
    st.session_state.filename = None


st.title("📑 ClauseIQ")

# -----------------------
# Upload stage
# -----------------------
if not st.session_state.analysis_done:
    st.markdown(
        """
        <style>
        .upload-container {
            display: flex;
            justify-content: center;
            align-items: center;
            flex-direction: column;
            margin-top: 5vh;
        }
        .upload-box {
            border: 2px dashed #6c63ff;
            border-radius: 15px;
            padding: 40px;
            width: 60%;
            text-align: center;
            background-color: #1e1e1e;
            transition: 0.3s ease;
        }
        .upload-box:hover {
            background-color: #2c2c2c;
            border-color: #8a85ff;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="upload-container">
            <h2>📤 Upload a Legal PDF Document</h2>
            <p>Drag & drop your file below, or click to browse</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Upload",
        type=["pdf"],
        label_visibility="collapsed",
    )
    
    

    if uploaded_file:
        st.success(f"📄 {uploaded_file.name} uploaded successfully.")
        analyze = st.button("🔍 Analyze Document")

        if analyze:
            # Save uploaded file
            st.session_state.filename = uploaded_file.name
            input_path = Path("data/input_docs") / uploaded_file.name
            input_path.parent.mkdir(parents=True, exist_ok=True)
            with open(input_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Pipeline: Extract → Entities → Risks → Summarize
            with st.spinner("⚙️ Analyzing document… please wait."):
                pages = extract_text_pages(str(input_path), use_ocr=True)
                text = "\n".join(pages)
                entities = extract_entities(text)
                risks = analyze_risks(text)
                json_result = summarize_document(text, entities, risks)

                st.session_state.json_result = json.loads(json_result)

                # Save PDF
                pdf_path = Path("data/output_docs") / "analysis_report.pdf"
                pdf_path.parent.mkdir(parents=True, exist_ok=True)
                save_json_to_pdf(json_result, filename=str(pdf_path))
                st.session_state.pdf_path = pdf_path

            # Index for RAG
            with st.spinner("🔍 Indexing document for Q&A…"):
                doc_id = str(uuid.uuid4())
                chunks = chunk_pages(pages, doc_id)
                upsert_chunks(chunks, doc_id)
                st.session_state.doc_id = doc_id

            st.session_state.analysis_done = True
            st.rerun()

# -----------------------
# Analysis stage
# -----------------------



else:
    # Guard against empty state during rerun
    if not st.session_state.json_result:
        st.stop()

    data = st.session_state.json_result
    
    if st.session_state.filename:
        st.success(f"📄 {st.session_state.filename} uploaded successfully.")
        
    st.divider()

    st.subheader(f"✅ AI-Enhanced Analysis:")

    st.markdown("### 📝 Summary")
    st.write(data.get("summary", "Not available"))

    st.markdown("### ⭐ Parties")
    st.write(data.get("parties", "Not available"))

    st.markdown("### 📅 Dates / Time")
    st.markdown(data.get("date/time", "Not available"), unsafe_allow_html=True)

    st.markdown("### 💰 Money / Penalties")
    st.write(data.get("money/penalties", "Not available"))

    st.markdown("### 📌 Obligations")
    st.write(data.get("obligations", "Not available"))

    st.markdown("### ⚖️ Risks")
    st.markdown(data.get("risks", "Not available"), unsafe_allow_html=True)

    st.markdown("### 💡 Suggestions")
    st.info(data.get("suggestion", "No suggestions provided."))

    with open(st.session_state.pdf_path, "rb") as f:
        pdf_bytes = f.read()

    st.download_button(
        label="⬇️ Download PDF Report",
        data=pdf_bytes,
        file_name="legal_analysis.pdf",
        mime="application/pdf",
    )

    st.button("🔄 Upload Another Document", on_click=reset_app)

    st.divider()

    # -----------------------
    # RAG Q&A section
    # -----------------------
    st.markdown("### 🤖 Ask Questions About This Document")

    for entry in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(entry["question"])
        with st.chat_message("assistant"):
            st.markdown(entry["answer"])
            st.divider()
            cols = st.columns(3)
            cols[0].caption(f"📄 Pages: {', '.join(str(p) for p in entry['pages']) or 'N/A'}")
            confidence = entry["confidence"]
            confidence_icon = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(confidence, "⚪")
            cols[1].caption(f"{confidence_icon} Confidence: {confidence.title()}")
            if entry["caveat"] and entry["caveat"].lower() != "none":
                cols[2].caption(f"⚠️ {entry['caveat']}")

    question = st.chat_input("Ask something about this document…")
    if question:
        with st.spinner("Thinking…"):
            result = answer(
                question=question,
                doc_id=st.session_state.doc_id,
            )
        st.session_state.chat_history.append({
            "question": question,
            "answer": result["answer"],
            "pages": result["relevant_pages"],
            "confidence": result["confidence"],
            "caveat": result["caveat"],
        })
        st.rerun()