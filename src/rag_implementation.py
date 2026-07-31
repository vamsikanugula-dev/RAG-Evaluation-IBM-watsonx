#!/usr/bin/env python
# coding: utf-8

# ### api key:jVMXkQMygbMBIAFnnp-UcXHzP8TcJXej-d8lJr6FCmn2
# ### projectid:e88ae624-24be-435f-bc21-d33e4b5326a7
# 

# In[1]:


get_ipython().run_line_magic('pip', 'install -U  langchain  langchain-community  langchain-text-splitters  langchain-core  langchain-ibm  langchain-huggingface  sentence-transformers  transformers  beautifulsoup4  faiss-cpu')
get_ipython().system('pip install -U langchain-experimental')


# In[3]:


import getpass
import requests

from bs4 import BeautifulSoup

from langchain_community.document_loaders import WebBaseLoader

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
)

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS

from langchain_ibm import WatsonxLLM

from ibm_watsonx_ai.metanames import (
    GenTextParamsMetaNames as GenParams,
)


# In[4]:


WATSONX_APIKEY = getpass.getpass(
    "Enter your watsonx.ai API Key: "
)

WATSONX_PROJECT_ID = getpass.getpass(
    "Enter your Project ID: "
)

URL = "https://eu-de.ml.cloud.ibm.com"


# In[5]:


from ibm_watsonx_ai import Credentials

credentials = Credentials(
    url=URL,
    api_key=WATSONX_APIKEY,
)

print("Credentials created successfully!")


# In[6]:


from langchain_ibm import WatsonxLLM
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

parameters = {
    GenParams.DECODING_METHOD: "greedy",
    GenParams.MAX_NEW_TOKENS: 256,
    GenParams.TEMPERATURE: 0.0,
}



llm = WatsonxLLM(
    model_id='mistralai/mistral-small-3-1-24b-instruct-2503',
    url=URL,
    apikey=WATSONX_APIKEY,
    project_id=WATSONX_PROJECT_ID,
    params=parameters,
)

print("LLM created successfully!")


# In[7]:


url = "https://www.ibm.com/new/announcements/ibm-granite-3-1-powerful-performance-long-context-and-more"

doc = WebBaseLoader(url).load()


# In[8]:


from langchain_huggingface import HuggingFaceEmbeddings
embeddings_model = HuggingFaceEmbeddings(model_name="ibm-granite/granite-embedding-30m-english")


# In[9]:


from transformers import AutoTokenizer
from langchain_text_splitters import CharacterTextSplitter

tokenizer = AutoTokenizer.from_pretrained(
    "ibm-granite/granite-3.1-8b-instruct"
)

text_splitter = CharacterTextSplitter.from_huggingface_tokenizer(
    tokenizer,
    separator="\n",
    chunk_size=1200,
    chunk_overlap=200,
)

fixed_size_chunks = text_splitter.create_documents([doc[0].page_content])
chunks = text_splitter.create_documents([doc[0].page_content])



# In[10]:


from langchain_core.documents import Document
proposition_prompt = """
Convert the following text into independent factual statements.

Rules:
- Each statement should contain only one fact.
- Do not add new information.
- Return only bullet points.

Text:
{text}
"""

propositions_chunks = []

for chunk in chunks:

    response = llm.invoke(
        proposition_prompt.format(
            text=chunk.page_content
        )
    )

    proposition_doc = Document(
        page_content=response,
        metadata={}
    )

    propositions_chunks.append(proposition_doc)
    


# In[11]:


from langchain_text_splitters import RecursiveCharacterTextSplitter



text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)

recursive_chunks = text_splitter.create_documents([doc[0].page_content])


# In[12]:


from langchain_classic.retrievers.parent_document_retriever import ParentDocumentRetriever
from langchain_community.vectorstores import FAISS
from langchain_classic.storage import InMemoryStore

parent_splitter = RecursiveCharacterTextSplitter(chunk_size=800,chunk_overlap=100)

child_splitter = RecursiveCharacterTextSplitter(chunk_size=250,chunk_overlap=50)

vectorstore = FAISS.from_texts(texts=["dummy"],embedding=embeddings_model)
vectorstore.delete(list(vectorstore.index_to_docstore_id.values()))

store = InMemoryStore()

retriever_chunks = ParentDocumentRetriever(vectorstore=vectorstore,docstore=store,child_splitter=child_splitter,parent_splitter=parent_splitter)


documents = [doc[0]]

# Split into parent documents
parent_docs = parent_splitter.split_documents(documents)

# Remove very small parent chunks (optional)
filtered_parent_docs = [
    d for d in parent_docs
    if len(d.page_content.strip()) >= 200
]

# Add to retriever
retriever_chunks.add_documents(filtered_parent_docs)


# In[13]:


from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker

# First split by size
recursive = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=200
)

initial_chunks = recursive.split_documents([doc[0]])

# Then semantic split
semantic = SemanticChunker(embeddings_model)

semantic_chunks = []

for chunk in initial_chunks:
    semantic_chunks.extend(
        semantic.create_documents([chunk.page_content])
    )


# In[14]:


final_semantic_chunks = [
    doc for doc in semantic_chunks
    if len(doc.page_content) >= 150
]


# In[15]:


vector_dbsemantic = FAISS.from_documents(
    final_semantic_chunks,
    embeddings_model
)
vector_dbfixed = FAISS.from_documents(
    fixed_size_chunks,
    embeddings_model
)
vector_dbrecursive = FAISS.from_documents(
    recursive_chunks,
    embeddings_model
)


print("FAISS vector database created successfully!")


# In[16]:


vector_dbparent = retriever_chunks.vectorstore
print("FAISS vector database created successfully!")


# In[17]:


vector_dbpropos = FAISS.from_documents(
    propositions_chunks,
    embeddings_model
)


# In[29]:


import os

os.makedirs("saved_rag", exist_ok=True)
print("sucessful")


# In[19]:


vector_dbfixed.save_local("faiss_index_fixed")

vector_dbsemantic.save_local("faiss_index_semantic")

vector_dbpropos.save_local("faiss_index_propos")

vector_dbrecursive.save_local("faiss_index_recursive")

vector_dbparent.save_local("faiss_index_parent")


# In[20]:


vector_dbfixed.save_local("saved_rag/faiss_index_fixed")

vector_dbsemantic.save_local("saved_rag/faiss_index_semantic")

vector_dbpropos.save_local("saved_rag/faiss_index_propos")

vector_dbrecursive.save_local("saved_rag/faiss_index_recursive")

vector_dbparent.save_local("saved_rag/faiss_index_parent")


# In[21]:


vector_dbsemantic = FAISS.load_local(
    "faiss_index_semantic",
    embeddings_model,
    allow_dangerous_deserialization=True
)
vector_dbfixed = FAISS.load_local(
    "faiss_index_fixed",
    embeddings_model,
    allow_dangerous_deserialization=True
)
vector_dbrecursive = FAISS.load_local(
    "faiss_index_recursive",
    embeddings_model,
    allow_dangerous_deserialization=True
)
vector_dbpropos = FAISS.load_local(
    "faiss_index_propos",
    embeddings_model,
    allow_dangerous_deserialization=True
)
vector_dbparent = FAISS.load_local(
    "faiss_index_parent",
    embeddings_model,
    allow_dangerous_deserialization=True
)


# In[22]:


import pickle
with open("saved_rag/documents.pkl", "wb") as f:
    pickle.dump(documents, f)

with open("saved_rag/fixed_size_chunks.pkl", "wb") as f:
    pickle.dump(fixed_size_chunks, f)
    
with open("saved_rag/final_semantics.pkl", "wb") as f:
    pickle.dump(final_semantic_chunks, f)

with open("saved_rag/propositions_chunks.pkl", "wb") as f:
    pickle.dump(propositions_chunks, f)   

with open("saved_rag/child_splitter1.pkl", "wb") as f:
    pickle.dump(child_splitter, f)

with open("saved_rag/parent_splitter1.pkl", "wb") as f:
    pickle.dump(parent_splitter, f) 
    


# In[23]:


import os
print(os.listdir("saved_rag"))


# In[ ]:





# In[ ]:




