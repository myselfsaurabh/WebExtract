import requests
from bs4 import BeautifulSoup
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.llms import Ollama
from langchain.embeddings import HuggingFaceEmbeddings




BASE_URL = "https://cloud.google.com"
DOCS_URL = "https://cloud.google.com/deployment-manager/docs/apis"

def get_links(url):
    """Extract all documentation links from the main API page."""
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to fetch {url}")
        return []

    soup = BeautifulSoup(response.text, "html.parser")

    # Extract all relevant API documentation links
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

# Get all API documentation links
doc_links = get_links(DOCS_URL)

# Crawl each page and extract content
api_docs = {}
for link in doc_links:
    print(f"Fetching: {link}")
    api_docs[link] = get_page_content(link)

# Save extracted data to a file
with open("deployment_manager_docs.txt", "w", encoding="utf-8") as f:
    for url, content in api_docs.items():
        f.write(f"### URL: {url}\n{content}\n\n")

os.environ['OPENAI_API_KEY'] = 'sk-proj-hphqPMtEIYnHYvvV9vDmqQc3KzTglv1AqE7gMr2tnEeenGW63gBV7Ea9rrOcqUzlzMbX0degEST3BlbkFJPUYqIYH/dQyPFPspb7I1UThl6zDBTn55XzHybaJyLFrr7EttoKF6szecC7plxpfbi3nHjv3kAw'
# A'
# Load extracted text
with open("deployment_manager_docs.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# Split text into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = splitter.split_text(raw_text)

# Store embeddings in ChromaDB
# vectorstore = Chroma.from_texts(chunks, embedding=OpenAIEmbeddings(), persist_directory="./gcp_deployment_vectorstore")
# Use Hugging Face Embeddings instead of OpenAI
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load vectorstore with local embeddings
vectorstore = Chroma(persist_directory="./gcp_deployment_vectorstore", embedding_function=embeddings)

vectorstore.persist()


# Load vector database
retriever = vectorstore.as_retriever()

# Use a local or remote model (change model to "mistral" if using Ollama)
# llm = ChatOpenAI(model="gpt-4")
llm = Ollama(model="mistral")

# Use from_chain_type to create the RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

# Example query
query = "write 10 test case for positive negative edge case scenario  if format steps expected for  v2 API deployment delete HTTP request"
response = qa_chain.run(query)

print(response)

