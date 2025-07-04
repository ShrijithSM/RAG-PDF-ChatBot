# Cell 9 (Optional: For Streamlit UI)
# Save this content as a Python file (e.g., `streamlit_app.py`) and run with `streamlit run streamlit_app.py`

import streamlit as st
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import os

# --- Configuration ---
PDF_FILE_PATH = 'note.pdf' # Make sure this PDF exists in the same directory
FAISS_INDEX_NAME = "pdf_rag_faiss_index.bin"
TEXTS_NAME = "pdf_rag_texts.npy"
LLM_MODEL_NAME = "google/flan-t5-small"

# --- Helper Functions (re-defined or imported from previous cells) ---
@st.cache_resource
def load_pdf(file_path):
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        st.success(f"Loaded PDF from: {file_path}")
        return text
    except Exception as e:
        st.error(f"Error loading PDF: {e}. Please ensure '{file_path}' exists and is a valid PDF.")
        return None

@st.cache_resource
def split_text_into_chunks(text, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.create_documents([text])
    st.success(f"Text split into {len(chunks)} chunks.")
    return chunks

@st.cache_resource
def get_embeddings_model():
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        st.success(f"SentenceTransformer model loaded successfully.")
        return model
    except Exception as e:
        st.error(f"Error loading SentenceTransformer model: {e}")
        return None

@st.cache_resource
def load_or_create_faiss_index(pdf_text_content, embedding_model):
    if os.path.exists(FAISS_INDEX_NAME) and os.path.exists(TEXTS_NAME):
        try:
            index = faiss.read_index(FAISS_INDEX_NAME)
            texts = np.load(TEXTS_NAME, allow_pickle=True).tolist()
            st.success(f"FAISS index and texts loaded from disk.")
            return index, texts
        except Exception as e:
            st.error(f"Error loading FAISS index or texts: {e}. Recreating...")
            # Fall through to creation if load fails
    if pdf_text_content and embedding_model:
        st.info("Creating FAISS index from scratch. This may take a moment...")
        chunks = split_text_into_chunks(pdf_text_content)
        chunk_contents = [chunk.page_content for chunk in chunks]
        embeddings = embedding_model.encode(chunk_contents, show_progress_bar=False)
        embeddings = np.array(embeddings).astype('float32')

        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)

        faiss.write_index(index, FAISS_INDEX_NAME)
        np.save(TEXTS_NAME, chunks) # Save Document objects directly
        st.success(f"FAISS index created and saved with {index.ntotal} vectors.")
        return index, chunks
    else:
        st.error("Cannot create FAISS index: PDF text content or embedding model not available.")
        return None, None

@st.cache_resource
def get_local_llm_generator():
    try:
        st.info(f"Loading local LLM: {LLM_MODEL_NAME}. This might take a while...")
        device = 0 if torch.cuda.is_available() else -1
        tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
        model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL_NAME,
                                                      torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
        generator = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95
        )
        st.success("Local LLM loaded successfully.")
        return generator
    except Exception as e:
        st.error(f"Error loading local LLM: {e}")
        st.warning("Consider using a smaller model or ensuring sufficient RAM/VRAM.")
        return None

def retrieve_chunks(query, embedding_model, faiss_index, indexed_texts, k=5):
    if embedding_model is None or faiss_index is None or indexed_texts is None:
        return []
    query_embedding = embedding_model.encode([query]).astype('float32')
    distances, indices = faiss_index.search(query_embedding, k)
    # Ensure indexed_texts contains actual Document objects or their page_content
    retrieved_chunks = [indexed_texts[i].page_content if hasattr(indexed_texts[i], 'page_content') else indexed_texts[i] for i in indices[0]]
    return retrieved_chunks

def generate_answer(query, retrieved_chunks, llm_generator):
    if llm_generator is None:
        return "Error: Language model not loaded."
    context = "\n".join(retrieved_chunks)
    prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
    try:
        response = llm_generator(prompt)
        answer = response[0]['generated_text']
        return answer
    except Exception as e:
        return f"An error occurred while generating the answer: {e}"

# --- Streamlit UI ---
st.set_page_config(page_title="Chat with PDF (RAG)", page_icon="📄")

st.title("📄 Chat with PDF (RAG)")
st.markdown("Ask questions about your PDF document using **Retrieval-Augmented Generation**.")
st.markdown("---")

# Initialize components
pdf_text_content = load_pdf(PDF_FILE_PATH)
embedding_model = get_embeddings_model()
faiss_index, indexed_texts = load_or_create_faiss_index(pdf_text_content, embedding_model)
llm_generator = get_local_llm_generator()

if not (pdf_text_content and embedding_model and faiss_index and indexed_texts and llm_generator):
    st.error("One or more essential components failed to load. Please check the console for errors and ensure your PDF is accessible.")
else:
    st.write(f"PDF loaded: `{os.path.basename(PDF_FILE_PATH)}` with {len(indexed_texts)} chunks.")

    # Chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask your question here..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching and generating answer..."):
                response = generate_answer(prompt,
                                           retrieve_chunks(prompt, embedding_model, faiss_index, indexed_texts),
                                           llm_generator)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

    st.markdown("---")
    st.markdown("Built with Embeddings, FAISS, and Local LLM (no OpenAI API).")