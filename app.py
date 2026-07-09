import streamlit as st
import webrag
import os

st.set_page_config(page_title="WebRAG Q&A", page_icon="🌐", layout="wide")

st.title("🌐 WebRAG: Grounded Q&A Bot")
st.markdown("Ask questions grounded in the live content of web pages. Uses LangChain, Web Loaders, Chroma, and OpenAI/GitHub Models.")

# Sidebar for URL input and indexing
st.sidebar.header("Source URLs")
url_input = st.sidebar.text_area("Enter URLs (one per line)", height=150)

if st.sidebar.button("Index / Refresh URLs"):
    urls = [url.strip() for url in url_input.split('\n') if url.strip()]
    if not urls:
        st.sidebar.error("Please enter at least one URL.")
    else:
        with st.spinner("Fetching and indexing..."):
            try:
                changes = webrag.index_urls(urls)
                st.sidebar.success("Indexing complete!")
                
                # Show changes
                st.subheader("Freshness Report")
                for change in changes:
                    status = change["status"]
                    url = change["url"]
                    if status == "new":
                        st.info(f"🆕 **New**: {url}")
                    elif status == "unchanged":
                        st.success(f"✅ **Unchanged**: {url}")
                    elif status == "changed":
                        st.warning(f"🔄 **Changed**: {url}")
                        with st.expander("View Diff"):
                            st.code(change["diff"], language="diff")
                            
            except Exception as e:
                st.sidebar.error(f"Error during indexing: {e}")

# Chat interface
st.header("Ask a Question")
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("Sources"):
                for doc in msg["sources"]:
                    st.markdown(f"**URL:** {doc.metadata.get('source')}")
                    st.text(doc.page_content[:200] + "...")

# Chat input
if prompt := st.chat_input("Ask a question based on the indexed URLs"):
    # Show user message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Get bot response
    with st.chat_message("assistant"):
        with st.spinner("Searching and generating answer..."):
            try:
                answer, docs = webrag.ask_question(prompt)
                st.markdown(answer)
                
                if docs:
                    with st.expander("Sources"):
                        for doc in docs:
                            st.markdown(f"**URL:** {doc.metadata.get('source')}")
                            st.text(doc.page_content[:200] + "...")
                
                # Save assistant message
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer,
                    "sources": docs
                })
            except Exception as e:
                st.error(f"Error answering question: {e}")
