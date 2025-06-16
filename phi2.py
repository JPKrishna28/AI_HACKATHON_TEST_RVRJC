import streamlit as st
import torch
import fitz  # PyMuPDF
import os
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- CONFIG ---
HF_TOKEN = os.environ["HF_TOKEN"]  # Taken from Hugging Face Space secrets

# Load tokenizer and model (replaced Gemma 2B with Phi-2)
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-2",
    torch_dtype=torch.float16,
    device_map="auto",
    token=HF_TOKEN
)

# Load sentence transformer for embeddings
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Sidebar for file upload and display
st.sidebar.title("📂 File Upload")
uploaded_file = st.sidebar.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])
file_display_area = st.sidebar.empty()


# Extract text from file
def extract_text(file):
    text = ""
    if file.type == "application/pdf":
        doc = fitz.open(stream=file.read(), filetype="pdf")
        for page in doc:
            text += page.get_text()
    elif file.type == "text/plain":
        text = file.read().decode("utf-8")
    return text


# Split into chunks
def split_into_chunks(text, chunk_size=500):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


# Create FAISS index
def create_faiss_index(chunks):
    embeddings = embedder.encode(chunks)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    return index, embeddings


# Retrieve top-k chunks
def retrieve_chunks(query, chunks, index, embeddings, k=5):
    query_embedding = embedder.encode([query])
    D, I = index.search(np.array(query_embedding), k)
    return [chunks[i] for i in I[0]]


# --- MAIN LOGIC ---
st.title("🔍 RAG App using 🤖 Phi-2")

if uploaded_file:
    file_display_area.success("✅ File uploaded successfully!")

    raw_text = extract_text(uploaded_file)
    chunks = split_into_chunks(raw_text)

    file_display_area.info(f"📚 Document split into {len(chunks)} chunks")
    file_display_area.text_area("📄 Extracted Document Text", raw_text, height=200)

    index, embeddings = create_faiss_index(chunks)

    # Chat-like interface
    st.markdown("### 💬 Chat with the Document")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_question = st.text_input("Ask something about the document:")

    if user_question:
        with st.spinner("Thinking..."):
            context = "\n".join(retrieve_chunks(user_question, chunks, index, embeddings))

            # Updated prompt for Phi-2's instruction style
            prompt = (
                f"Instruction: Answer the following question using only the context provided. "
                f"Extract specific information directly from the context when available. "
                f"If the answer is not in the context, respond with 'Information not found.'\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {user_question}\n\n"
                f"Answer: "
            )

            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    input_ids,
                    max_new_tokens=512,  # Keep using max_new_tokens as fixed before
                    num_return_sequences=1,
                    temperature=0.2,  # Lower temperature for more focused answers
                    do_sample=True,  # Enable sampling for more natural responses
                    top_p=0.9,  # Add top_p sampling for better quality
                )

            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract the answer part - adapt based on Phi-2's output format
            if "Answer:" in generated_text:
                answer = generated_text.split("Answer:")[-1].strip()
            else:
                answer = generated_text.replace(prompt, "").strip()

        # Append interaction to chat history
        st.session_state.chat_history.append({"question": user_question, "answer": answer})

    # Display chat history
    for chat in st.session_state.chat_history:
        st.markdown(f"**You:** {chat['question']}")
        st.markdown(f"**Phi-2:** {chat['answer']}")