import requests
import streamlit as st

API_BASE = "http://127.0.0.1:8000"

st.set_page_config(page_title="QueryPilot", layout="wide")
st.markdown(
    "<h3 style='text-align: left; margin-top: -60px; postion: fixed'>QueryPilot</h3>",
    unsafe_allow_html=True
)
if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = None

if "selected_model" not in st.session_state:
    st.session_state.selected_model = "llama-3.3-70b-versatile"

st.markdown("""
    <div style="
        display: flex;
        justify-content: center;
        align-items: center;
        height: 60vh;
    ">
        <h1>Where should we begin?</h1>
    </div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.subheader("Select Model")
    st.session_state.selected_model = st.selectbox(
        "Model",
        ["llama-3.3-70b-versatile", "gpt-4o"],
        index=0,
        label_visibility="collapsed"
    )

    st.subheader("Upload Document")
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["pdf", "docx", "xlsx", "html"]
    )

    if uploaded_file is not None and st.button("Upload File"):
        files = {
            "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
        }
        try:
            with st.spinner("Uploading and indexing document..."):
                res = requests.post(f"{API_BASE}/upload-doc", files=files, timeout=120)

            if res.status_code == 200:
                st.success("File uploaded successfully.")
                st.rerun()
            else:
                st.error(f"Upload failed: {res.status_code} - {res.text}")
        except Exception as e:
            st.error(f"Upload error: {e}")

    st.subheader("Uploaded Documents")

    docs = []
    try:
        res = requests.get(f"{API_BASE}/list-docs", timeout=30)
        if res.status_code == 200:
            data = res.json()
            docs = data if isinstance(data, list) else data.get("documents", [])
            print(docs, "checking>>>>>>>>>>>>>>")
        else:
            st.warning("Could not load document list.")
    except Exception as e:
        st.error(f"Document list error: {e}")

    doc_options = {}

    if docs:
        for d in docs:
            if isinstance(d, dict):
                file_id = d.get("id")
                filename = d.get("filename")
                if file_id is not None and filename:
                    st.write(filename)
                    doc_options[filename] = file_id

        selected_doc_name = st.selectbox(
            "Select a document to delete",
            options=list(doc_options.keys())
        )

        if selected_doc_name and st.button("Delete Selected Document"):
            file_id = doc_options[selected_doc_name]
            try:
                with st.spinner("Deleting document..."):
                    res = requests.post(
                        f"{API_BASE}/delete-doc",
                        json = {"file_id": file_id},
                        timeout = 60
                    )

                if res.status_code == 200:
                    st.success("Document deleted successfully.")
                    st.rerun()
                else:
                    st.error(f"Delete failed: {res.status_code} - {res.text}")
            except Exception as e:
                st.error(f"Delete error: {e}")
    else:
        st.info("No uploaded documents found.")

    if st.button("New Chat"):
        st.session_state.messages = []
        st.session_state.session_id = None
        st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Ask Your Query :)")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    payload = {
        "question": prompt,
        "session_id": st.session_state.session_id,
        "model": st.session_state.selected_model,
    }

    try:
        with st.spinner("Thinking..."):
            res = requests.post(f"{API_BASE}/chat", json=payload, timeout=120)

        if res.status_code == 200:
            data = res.json()
            answer = data.get("answer", "No answer returned.")
            st.session_state.session_id = data.get(
                "session_id",
                st.session_state.session_id
            )
        else:
            answer = f"API error: {res.status_code} - {res.text}"
    except Exception as e:
        answer = f"Request failed: {e}"

    st.session_state.messages.append({"role": "assistant", "content": answer})

    with st.chat_message("assistant"):
        st.markdown(answer)