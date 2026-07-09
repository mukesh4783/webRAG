import os
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

import json
import hashlib
import difflib
import shutil
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

STATE_FILE = "page_states.json"
CHROMA_DIR = "./chroma_db"

def get_token():
    # Reads token from .env file securely
    token = os.environ.get("OPENAI_API_KEY", os.environ.get("GITHUB_TOKEN", ""))
    if not token:
        raise ValueError("Please set GITHUB_TOKEN in your .env file.")
    return token

def get_llm():
    return ChatOpenAI(
        model="gpt-4o-mini",
        api_key=get_token(),
        base_url="https://models.inference.ai.azure.com",
    )

def get_embeddings():
    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=get_token(),
        base_url="https://models.inference.ai.azure.com",
    )

def load_states():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_states(states):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(states, f)

def compute_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def index_urls(urls):
    """
    Downloads the provided URLs, checks if they changed, updates states, 
    and saves to vector DB.
    """
    states = load_states()
    
    # Load pages
    loader = WebBaseLoader(urls)
    docs = loader.load()
    
    changes = []
    
    for doc in docs:
        url = doc.metadata.get("source", "unknown_url")
        text = doc.page_content
        current_hash = compute_hash(text)
        
        if url in states:
            old_hash = states[url].get("hash")
            if current_hash != old_hash:
                old_text = states[url].get("text", "")
                
                # Compute diff
                diff = list(difflib.unified_diff(
                    old_text.splitlines(),
                    text.splitlines(),
                    fromfile="Old",
                    tofile="New",
                    lineterm=""
                ))
                
                # We only want to show a small snippet of the diff
                diff_snippet = "\n".join(diff[:30])
                if len(diff) > 30:
                    diff_snippet += "\n... [diff truncated]"
                    
                changes.append({
                    "url": url, 
                    "status": "changed",
                    "diff": diff_snippet
                })
                states[url] = {"hash": current_hash, "text": text}
            else:
                changes.append({
                    "url": url,
                    "status": "unchanged",
                    "diff": ""
                })
        else:
            changes.append({
                "url": url, 
                "status": "new",
                "diff": "Initial indexing of this URL."
            })
            states[url] = {"hash": current_hash, "text": text}
            
    # Process for Vector DB
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    # Recreate the Chroma DB to avoid duplicates easily. 
    # On Windows, shutil.rmtree often fails due to file locks, so we tell Chroma to delete the collection.
    if os.path.exists(CHROMA_DIR):
        try:
            old_db = Chroma(persist_directory=CHROMA_DIR, embedding_function=get_embeddings())
            old_db.delete_collection()
        except Exception as e:
            print(f"Failed to clear old collection: {e}")
            
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=get_embeddings(), 
        persist_directory=CHROMA_DIR
    )
    
    save_states(states)
    return changes

def ask_question(query):
    if not os.path.exists(CHROMA_DIR):
        return "Index is empty. Please index some URLs first.", []
        
    vectorstore = Chroma(
        persist_directory=CHROMA_DIR, 
        embedding_function=get_embeddings()
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    
    system_prompt = (
        "Answer the question based ONLY on the following context. "
        "If the answer is not contained in the context, say "
        "'I couldn't find that information in the indexed pages.'\n\n"
        "{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(get_llm(), prompt)
    chain = create_retrieval_chain(retriever, question_answer_chain)
    
    result = chain.invoke({"input": query})
    return result["answer"], result["context"]
