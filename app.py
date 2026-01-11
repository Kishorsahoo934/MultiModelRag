import streamlit as st
import fitz
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai
import open_clip
import torch
import uuid
from PIL import Image
import io

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(page_title="PS-Compliant Multimodal RAG", layout="wide")
st.title("PS-Compliant Multimodal RAG System")

# =============================
# API CONFIG
# =============================
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# =============================
# LOAD MODELS
# =============================
@st.cache_resource
def load_models():
    text_model = SentenceTransformer("all-MiniLM-L6-v2")
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    clip_model.eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    return text_model, clip_model, preprocess, tokenizer

text_embedder, clip_model, clip_preprocess, clip_tokenizer = load_models()

# =============================
# CHROMADB COLLECTIONS
# =============================
def get_collections():
    client = chromadb.PersistentClient(
        path="./chroma_db",
        settings=Settings()
    )
    return (
        client.get_or_create_collection("text_rag"),
        client.get_or_create_collection("image_rag")
    )

text_collection, image_collection = get_collections()

# =============================
# PDF INGESTION
# =============================
def extract_pdf_pages(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []
    for i, page in enumerate(doc, start=1):
        text = page.get_text().strip()
        if text:
            pages.append({"text": text, "page": i})
    return pages

def chunk_pages(pages):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = []
    for p in pages:
        for part in splitter.split_text(p["text"]):
            chunks.append({"content": part, "page": p["page"]})
    return chunks

def index_pdf(chunks, filename):
    embeddings = text_embedder.encode([c["content"] for c in chunks])
    for i, c in enumerate(chunks):
        text_collection.add(
            documents=[c["content"]],
            embeddings=[embeddings[i].tolist()],
            metadatas=[{
                "modality": "text",
                "source": filename,
                "page": c["page"]
            }],
            ids=[str(uuid.uuid4())]
        )

# =============================
# IMAGE INGESTION
# =============================
def index_image(image_bytes, filename):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = clip_preprocess(image).unsqueeze(0)

    with torch.no_grad():
        emb = clip_model.encode_image(tensor).cpu().numpy()[0]

    image_collection.add(
        documents=[f"Image from {filename}"],
        embeddings=[emb.tolist()],
        metadatas=[{
            "modality": "image",
            "source": filename
        }],
        ids=[str(uuid.uuid4())]
    )

# =============================
# INTENT ANALYSIS
# =============================
def analyze_intent(question):
    q = question.lower()
    image_words = ["image", "diagram", "figure", "photo", "visual"]
    text_words = ["explain", "define", "page", "pdf", "document"]

    return {
        "use_text": not any(w in q for w in image_words),
        "use_image": not any(w in q for w in text_words)
    }

# =============================
# RETRIEVAL
# =============================
def retrieve(question, k=3, retry=False):
    intent = analyze_intent(question)
    contexts = []

    if intent["use_text"]:
        emb = text_embedder.encode([question]).tolist()
        res = text_collection.query(
            query_embeddings=emb,
            n_results=k * (2 if retry else 1)
        )
        for d, m in zip(res["documents"][0], res["metadatas"][0]):
            contexts.append((m, d))

    if intent["use_image"]:
        emb = clip_model.encode_text(
            clip_tokenizer([question])
        ).detach().cpu().numpy().tolist()

        res = image_collection.query(
            query_embeddings=emb,
            n_results=k * (2 if retry else 1)
        )
        for d, m in zip(res["documents"][0], res["metadatas"][0]):
            contexts.append((m, d))

    return contexts

# =============================
# EVIDENCE CONFIDENCE
# =============================
def assess_evidence(contexts):
    score = 0
    pages = set()

    for meta, _ in contexts:
        if meta["modality"] == "text":
            score += 20
            pages.add((meta["source"], meta["page"]))
        else:
            score += 10

    score = min(score, 100)

    return {
        "score": score,
        "sources": sorted(pages)
    }

# =============================
# SAFE RETRIEVAL
# =============================
def safe_retrieve(question):
    try:
        contexts = retrieve(question)
        if assess_evidence(contexts)["score"] < 40:
            contexts = retrieve(question, retry=True)
        return contexts
    except Exception:
        return []

# =============================
# ANSWER GENERATION (FULL ANSWERS)
# =============================
def ask_gemini(contexts, question):
    if not contexts:
        return {
            "answer": "Not found in the documents. Required evidence is missing.",
            "confidence": 0,
            "citations": []
        }

    assessment = assess_evidence(contexts)

    evidence_text = "\n\n".join(
        content for meta, content in contexts
        if meta["modality"] == "text"
    )

    prompt = f"""
You are a Multimodal Retrieval-Augmented Generation system.

TASK:
Write a COMPLETE, well-explained answer using ONLY the evidence.
You may summarize and combine multiple points.

RULES:
- Do not introduce new facts
- Do not copy a single sentence blindly
- Write a full paragraph explanation

EVIDENCE:
{evidence_text}

QUESTION:
{question}
"""

    model = genai.GenerativeModel("gemini-2.5-flash")
    answer = model.generate_content(prompt).text.strip()

    citations = [
        f"{src} (Page {pg})"
        for src, pg in assessment["sources"]
    ]

    return {
        "answer": answer,
        "confidence": assessment["score"],
        "citations": citations
    }

# =============================
# UI
# =============================
uploaded_files = st.file_uploader(
    "Upload PDFs and Images",
    type=["pdf", "png", "jpg", "jpeg"],
    accept_multiple_files=True
)

question = st.text_input("Ask a question across all uploaded content")

if uploaded_files:
    if st.button("Index Files"):
        for f in uploaded_files:
            if f.type == "application/pdf":
                pages = extract_pdf_pages(f.read())
                chunks = chunk_pages(pages)
                index_pdf(chunks, f.name)
            else:
                index_image(f.read(), f.name)
        st.success("Files indexed successfully")

if question:
    with st.spinner("Retrieving and reasoning..."):
        contexts = safe_retrieve(question)
        result = ask_gemini(contexts, question)

        st.subheader("Answer")
        st.write(result["answer"])

        st.subheader("Confidence Score")
        st.progress(result["confidence"] / 100)
        st.write(f"{result['confidence']}% confidence based on retrieved evidence")

        st.subheader("Sources")
        for cite in result["citations"]:
            st.markdown(f"- {cite}")
