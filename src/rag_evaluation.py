#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('pip', 'install -U  langchain  langchain-community  langchain-text-splitters  langchain-core  langchain-ibm  langchain-huggingface  sentence-transformers  transformers  beautifulsoup4  faiss-cpu')
get_ipython().system('pip install -U langchain-experimental')
get_ipython().run_line_magic('pip', 'install ragas datasets')


# In[2]:


import getpass
import requests

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS

from langchain_ibm import WatsonxLLM

from ibm_watsonx_ai.metanames import (
    GenTextParamsMetaNames as GenParams,
)


# In[3]:


WATSONX_APIKEY = getpass.getpass(
    "Enter your watsonx.ai API Key: "
)

WATSONX_PROJECT_ID = getpass.getpass(
    "Enter your Project ID: "
)

URL = "https://eu-de.ml.cloud.ibm.com"


# In[4]:


from ibm_watsonx_ai import Credentials

credentials = Credentials(
    url=URL,
    api_key=WATSONX_APIKEY,
)

print("Credentials created successfully!")


# In[5]:


from langchain_ibm import ChatWatsonx
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

parameters = {
    GenParams.MAX_NEW_TOKENS: 2048,
    GenParams.TEMPERATURE: 0.0,
}

llm = ChatWatsonx(
    model_id="mistralai/mistral-small-3-1-24b-instruct-2503",
    url=URL,
    apikey=WATSONX_APIKEY,
    project_id=WATSONX_PROJECT_ID,
    params=parameters,
)


# In[6]:


from langchain_huggingface import HuggingFaceEmbeddings
embeddings_model = HuggingFaceEmbeddings(model_name="ibm-granite/granite-embedding-30m-english")


# In[7]:


import pickle

with open("saved_rag/documents.pkl", "rb") as f:
    documents = pickle.load(f)

with open("saved_rag/fixed_size_chunks.pkl", "rb") as f:
    fixed_size_chunks = pickle.load(f)

with open("saved_rag/final_semantics.pkl", "rb") as f:
    semantic_chunks = pickle.load(f)

with open("saved_rag/propositions_chunks.pkl", "rb") as f:
    propositions_chunks = pickle.load(f)

with open("saved_rag/child_splitter1.pkl", "rb") as f:
    child_splitter = pickle.load(f)

with open("saved_rag/parent_splitter1.pkl", "rb") as f:
    parent_splitter = pickle.load(f)


# In[8]:


from langchain_community.vectorstores import FAISS
vector_dbfixed = FAISS.load_local(
    "saved_rag/faiss_index_fixed",
    embeddings_model,
    allow_dangerous_deserialization=True
)

vector_dbsemantic = FAISS.load_local(
    "saved_rag/faiss_index_semantic",
    embeddings_model,
    allow_dangerous_deserialization=True
)

vector_dbpropos = FAISS.load_local(
    "saved_rag/faiss_index_propos",
    embeddings_model,
    allow_dangerous_deserialization=True
)

vector_dbrecursive = FAISS.load_local(
    "saved_rag/faiss_index_recursive",
    embeddings_model,
    allow_dangerous_deserialization=True
)

vector_dbparent = FAISS.load_local(
    "saved_rag/faiss_index_parent",
    embeddings_model,
    allow_dangerous_deserialization=True
)


# In[10]:


from langchain_classic.retrievers.parent_document_retriever import ParentDocumentRetriever
from langchain_classic.storage import InMemoryStore

store = InMemoryStore()
vectorstore = vector_dbparent
retriever_chunks = ParentDocumentRetriever(vectorstore=vectorstore,docstore=store,child_splitter=child_splitter,parent_splitter=parent_splitter)
retriever_chunks.add_documents(documents)


# In[11]:


eval_questions = [
    "What improvements does Granite 3.1 provide?",
    "What is Granite Embedding used for?",
    "What is the context length of Granite 3.1?"
]


ground_truth = [
    "Granite 3.1 provides performance improvements, 128K context length, Granite Embedding models, and better hallucination detection.",
    "Granite Embedding models are used for semantic search, vector search and RAG applications.",
    "Granite 3.1 models support a 128K token context window."
]


# In[14]:


def run_rag(vector_db, query):

    docs = vector_db.similarity_search(
        query,
        k=3
     )

    context = [doc.page_content for doc in docs]

   

    prompt = f"""
    Answer only using the context.

    Context:
    {context}

    Question:
    {query}

    Answer:
    """

    answer = llm.invoke(prompt)

    return {
        "question":query,
        "answer":answer,
        "contexts":context
    }

    
def parent_run_rag(vector_db, query):

    child_docs = vector_db.similarity_search(
        query,
        k=3
     )
    parent_ids = list(set([doc.metadata["doc_id"] for doc in child_docs]))

    parent_docs = retriever_chunks.docstore.mget(parent_ids)
    
    context = [
        doc.page_content for doc in parent_docs if doc is not None
      ]

   

    prompt = f"""
    Answer only using the context.

    Context:
    {context}

    Question:
    {query}

    Answer:
    """

    answer = llm.invoke(prompt)

    return {
        "question":query,
        "answer":answer,
        "contexts":context
    }    


# In[27]:


semantic_results=[]
for q in eval_questions:
    
    result = run_rag(vector_dbsemantic,q)
    semantic_results.append(result)


propos_results=[]
for q in eval_questions:

    result = run_rag(vector_dbpropos ,q)

    propos_results.append(result)


parent_results = []
for q in eval_questions:

    result = parent_run_rag(vector_dbparent,q)

    parent_results.append(result)    

fixed_results = []

for q in eval_questions:

    result = run_rag(vector_dbfixed,q)

    fixed_results.append(result) 
    

recursive_results = []


for q in eval_questions:

    result = run_rag(vector_dbrecursive,q)

    recursive_results.append(result) 

  
    


# In[28]:


from datasets import Dataset


semantic_dataset = Dataset.from_dict(
  {
    "question":[x["question"] for x in semantic_results],

    "answer":[x["answer"].content for x in semantic_results],

    "contexts":[x["contexts"] for x in semantic_results],

    "ground_truth":ground_truth
  }
   )

propos_dataset = Dataset.from_dict(
{
    "question":[x["question"] for x in propos_results],

    "answer":[x["answer"].content for x in propos_results],

    "contexts":[x["contexts"] for x in propos_results],

    "ground_truth":ground_truth
}
)

parent_dataset = Dataset.from_dict(
{
    "question":[x["question"] for x in parent_results],

    "answer":[x["answer"].content for x in parent_results],

    "contexts":[x["contexts"] for x in parent_results],

    "ground_truth":ground_truth
}
)

fixed_dataset = Dataset.from_dict(
{
    "question":[x["question"] for x in fixed_results],

    "answer":[x["answer"].content for x in fixed_results],

    "contexts":[x["contexts"] for x in fixed_results],

    "ground_truth":ground_truth
}
)

recursive_dataset = Dataset.from_dict(
{
    "question":[x["question"] for x in recursive_results],

    "answer":[x["answer"].content for x in recursive_results],

    "contexts":[x["contexts"] for x in recursive_results],

    "ground_truth":ground_truth
}
)



# In[17]:


print(parent_dataset.features)


# In[18]:


for i, c in enumerate(parent_dataset[0]["contexts"]):
    print(i, len(c))


# In[19]:


get_ipython().run_line_magic('pip', 'install langchain-google-vertexai')


# In[20]:


import sys
import langchain_google_vertexai

sys.modules["langchain_community.chat_models.vertexai"] = langchain_google_vertexai


# In[22]:


from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

evaluator_llm = LangchainLLMWrapper(llm)
evaluator_embeddings = LangchainEmbeddingsWrapper(embeddings_model)


# In[29]:


from ragas import evaluate

from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)


semantic_score = evaluate(
    semantic_dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall
    ],
    llm=evaluator_llm,
    embeddings=evaluator_embeddings
)


print(semantic_score)


# In[30]:


propos_score = evaluate(
    propos_dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall
    ],
    llm=evaluator_llm,
    embeddings=evaluator_embeddings
)
print(propos_score)


# In[31]:


parent_score = evaluate(
    parent_dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall
    ],
    llm=evaluator_llm,
    embeddings=evaluator_embeddings
)
print(parent_score)


# In[32]:


fixed_score = evaluate(
    fixed_dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall
    ],
    llm=evaluator_llm,
    embeddings=evaluator_embeddings
)
print(fixed_score)


# In[33]:


recursive_score = evaluate(
    recursive_dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall
    ],
    llm=evaluator_llm,
    embeddings=evaluator_embeddings
)
print(recursive_score)


# In[ ]:




