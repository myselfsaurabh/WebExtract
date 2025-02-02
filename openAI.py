import requests
from bs4 import BeautifulSoup
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM

# 🔹 Base URLs
BASE_URL = "https://cloud.google.com"
DOCS_URL = "https://cloud.google.com/deployment-manager/docs/apis"
# DOCS_URL = "https://cloud.google.com/deployment-manager/docs/reference/v2beta"


def get_links(url):
    """Extract all documentation links from the main API page."""
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to fetch {url}")
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    
    # Extract relevant API documentation links
    links = []
    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        if href.startswith("/deployment-manager/docs/"):
            links.append(BASE_URL + href)

    return list(set(links))  # Remove duplicates

def get_page_content(url):
    """Extract text content from a given page."""
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to fetch {url}")
        return ""

    soup = BeautifulSoup(response.text, "html.parser")

    # Extract text from paragraphs and code blocks
    content = []
    for tag in soup.find_all(["p", "pre", "code"]):
        content.append(tag.get_text(strip=True))

    return "\n".join(content)

# 🔹 Step 1: Get all API documentation links
doc_links = get_links(DOCS_URL)

# 🔹 Step 2: Crawl each page and extract content
api_docs = {}
for link in doc_links:
    print(f"Fetching: {link}")
    api_docs[link] = get_page_content(link)

# 🔹 Step 3: Save extracted data to a file
with open("deployment_manager_docs.txt", "w", encoding="utf-8") as f:
    for url, content in api_docs.items():
        f.write(f"### URL: {url}\n{content}\n\n")

# 🔹 Step 4: Load extracted text
with open("deployment_manager_docs.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# 🔹 Step 5: Split text into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = splitter.split_text(raw_text)

# 🔹 Step 6: Store embeddings in ChromaDB (Fixing persistence issue)
embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create and persist the vectorstore
vectorstore = Chroma.from_texts(chunks, embedding=embedding_function, persist_directory="./gcp_deployment_vectorstore")
vectorstore.persist()

# 🔹 Step 7: Load vector database properly
retriever = vectorstore.as_retriever()

# 🔹 Step 8: Load Local Mistral Model
llm = OllamaLLM(model="mistral:instruct")

# 🔹 Step 9: Create the RAG Chain
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

# 🔹 Step 10: Query the RAG system
query = "Write 10 test cases for positive, negative, and edge case scenarios for a v2 API deployment delete HTTP request"
response = qa_chain.invoke(query)

print("\n🔹 AI Response:\n", response)