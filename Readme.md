 # ?? RAG Chunking Strategy Evaluation using IBM watsonx.ai

A comprehensive comparison of multiple document chunking strategies for Retrieval-Augmented Generation (RAG) using IBM watsonx.ai, LangChain, vector databases, and the Ragas evaluation framework.

---

# ?? Overview

Retrieval-Augmented Generation (RAG) improves the quality of Large Language Model (LLM) responses by retrieving relevant information from external knowledge sources before generating an answer. One of the most important components of a RAG pipeline is **document chunking**, which determines how source documents are divided before indexing and retrieval.

Different chunking strategies preserve contextual information differently, directly influencing retrieval quality and the accuracy of generated responses.

This project evaluates and compares five document chunking strategies using the IBM watsonx.ai platform. Each strategy is implemented and evaluated under identical experimental conditions using the same dataset, embedding model, retrieval configuration, and evaluation metrics to ensure a fair comparison.

The project also demonstrates the complete lifecycle of a Retrieval-Augmented Generation pipeline—from document loading and chunk generation to vector indexing, response generation, and automated evaluation using the Ragas framework.

---

# ? Key Features

- Comparative evaluation of five document chunking strategies.
- End-to-end Retrieval-Augmented Generation (RAG) pipeline using IBM watsonx.ai.
- LangChain-based document loading and orchestration.
- Independent vector databases for each chunking strategy.
- IBM Granite embedding and language models.
- Hugging Face Dataset integration for evaluation.
- Automated evaluation using the Ragas framework.
- Reusable serialized artifacts (`.pkl`) for faster experimentation.
- Performance comparison using multiple RAG evaluation metrics.

---

# ? Problem Statement

Document chunking significantly influences the performance of Retrieval-Augmented Generation systems. Poor chunking may lead to fragmented context, incomplete retrieval, and lower response quality.

Although several chunking techniques have been proposed, their effectiveness often depends on the characteristics of the dataset and retrieval pipeline.

This project investigates how different chunking strategies affect retrieval quality and response generation when applied to IBM watsonx.ai technical documentation.

---

# ?? Objectives

The primary objectives of this project are:

- Build an end-to-end Retrieval-Augmented Generation (RAG) pipeline.
- Implement multiple document chunking strategies.
- Compare the performance of different chunking techniques.
- Evaluate retrieval quality using standardized RAG evaluation metrics.
- Identify the most effective chunking strategy for the selected dataset.
- Analyze how chunking impacts retrieval and answer generation.

---

# ??? System Architecture

The project follows the workflow shown below:

IBM watsonx.ai Documentation

?

Document Loading (LangChain)

?

Embedding Model Initialization

?

Chunking Strategy

?

Vector Database Construction

?

Retriever

?

IBM Granite Large Language Model

?

Generated Response

?

Hugging Face Dataset Creation

?

Ragas Evaluation

---

# ??? Technology Stack

The project integrates multiple tools and frameworks to build, evaluate, and compare Retrieval-Augmented Generation (RAG) pipelines. Each technology serves a specific purpose within the overall workflow.

| Technology | Purpose |
|------------|---------|
| **Python** | Primary programming language used for implementing the RAG pipeline and evaluation workflow. |
| **IBM watsonx.ai** | Provides the embedding models, Large Language Models (LLMs), and runtime environment. |
| **LangChain** | Orchestrates document loading, chunking, vector database integration, retrieval, and response generation. |
| **Hugging Face Datasets** | Organizes retrieved contexts, generated answers, and reference answers into a standardized dataset for evaluation. |
| **FAISS / ChromaDB** | Stores vector embeddings and performs semantic similarity search. |
| **IBM Granite Embedding Model** | Converts document chunks into dense vector embeddings. |
| **IBM Granite LLM** | Generates responses using the retrieved contextual information. |
| **Ragas** | Evaluates retrieval and response quality using multiple RAG-specific metrics. |
| **Jupyter Notebook** | Interactive development environment used for experimentation and implementation. |
| **Pickle (.pkl)** | Stores intermediate artifacts to reduce repeated computation during experimentation. |

---

## ?? Why These Technologies?

### IBM watsonx.ai

IBM watsonx.ai serves as the primary AI platform for this project. It provides access to IBM Granite foundation models, embedding models, and a managed runtime environment for executing the complete Retrieval-Augmented Generation pipeline.

---

### LangChain

LangChain acts as the orchestration framework connecting every stage of the pipeline. It simplifies document loading, chunking, embedding generation, retriever creation, prompt construction, and interaction with Large Language Models.

---

### Hugging Face Datasets

The evaluation framework requires data to be organized in a structured format. Hugging Face Datasets stores user queries, retrieved contexts, generated responses, and ground truth answers before they are evaluated using Ragas.

---

### Vector Database

The vector database stores embeddings generated from each chunking strategy and enables semantic similarity search during retrieval.

To ensure a fair comparison, each chunking strategy maintains an independent vector database.

---

### Ragas

Ragas provides an automated evaluation framework specifically designed for Retrieval-Augmented Generation systems.

It measures multiple aspects of retrieval and answer quality using standardized metrics such as Faithfulness, Answer Relevancy, Context Precision, and Context Recall.

---

### Pickle Serialization

Several intermediate outputs—including chunk collections, retrieval artifacts, and evaluation datasets—are serialized into `.pkl` files.

This allows expensive computations to be reused, significantly reducing execution time during repeated experiments.

---

# ?? Dataset & Data Collection

## Overview

The knowledge source used in this project is the **IBM watsonx.ai Documentation**, which contains technical documentation related to IBM's AI platform, foundation models, prompt engineering, vector indexing, retrieval, and Retrieval-Augmented Generation (RAG).

The documentation serves as the external knowledge base for building and evaluating the RAG pipeline.

---

## Why IBM watsonx.ai Documentation?

The documentation was selected because:

- It contains well-structured technical information.
- It includes concepts requiring contextual retrieval.
- It represents real-world enterprise documentation.
- It is suitable for evaluating different document chunking strategies.
- It contains interconnected topics that benefit from Retrieval-Augmented Generation.

---

## Document Collection

The documentation is collected directly from IBM watsonx.ai documentation pages and loaded into the pipeline using LangChain document loaders.

Each webpage is converted into a LangChain `Document` object containing:

- Page Content
- Metadata
- Source Information

These document objects become the input for all chunking strategies.

---

## Document Preparation

The IBM watsonx.ai documentation is converted into LangChain `Document` objects without performing additional preprocessing such as stemming, lemmatization, stop-word removal, or text normalization.

Using the same unmodified source documents across every experiment ensures that differences in evaluation metrics arise from the chunking strategy rather than document preprocessing.

---

## Dataset Flow

IBM watsonx.ai Documentation

?

Document Loader

?

LangChain Document Objects

?

Chunking Strategy

?

Embedding Generation

?

Vector Database

?

Retriever

?

LLM

?

Evaluation

---

# ?? IBM watsonx.ai Environment Setup

This project is built and executed using the IBM watsonx.ai platform. Before running the project, a watsonx.ai project must be created and configured with the required credentials.

---

## Prerequisites

The following credentials are required:

- IBM Cloud API Key
- IBM watsonx.ai Project ID
- IBM watsonx.ai Service URL

These credentials authenticate requests to IBM's embedding models and Large Language Models.

---

## Authentication Workflow

The project authenticates with IBM watsonx.ai by initializing the SDK using the project credentials.

Authentication enables access to:

- IBM Granite Foundation Models
- IBM Granite Embedding Models
- Model inference services
- Runtime execution environment

Once authenticated, the same connection is reused throughout the project.

---

## Environment Configuration

The required credentials should be stored securely as environment variables or configuration variables instead of being hardcoded into the source code.

Typical configuration includes:

- API Key
- Project ID
- Service URL

This improves security, portability, and deployment flexibility.

---

## Runtime Environment

Experiments were executed using the IBM watsonx.ai runtime environment, which provides the infrastructure required for embedding generation, Large Language Model inference, and evaluation workflows.

---

# ?? Implementation Details

The project is implemented as a modular Retrieval-Augmented Generation (RAG) pipeline. Each stage is designed to be reusable, allowing different chunking strategies to be evaluated independently while maintaining identical retrieval and evaluation settings.

The implementation follows the workflow shown below:

IBM watsonx.ai Documentation

?

Document Loading

?

Embedding Model Initialization

?

Chunking Strategy

?

Vector Database Construction

?

Retriever

?

Large Language Model

?

Dataset Creation

?

Ragas Evaluation

---

## ?? Document Loading

The IBM watsonx.ai documentation is used as the knowledge source for the RAG pipeline.

The documentation is loaded using LangChain document loaders and converted into LangChain `Document` objects.

Each `Document` contains:

- Document content (`page_content`)
- Metadata
- Source information

Using the same source documents across every experiment ensures consistency and enables a fair comparison between chunking techniques.

---

## ?? Embedding Model Initialization

An IBM Granite Embedding Model is initialized through IBM watsonx.ai.

The embedding model serves two purposes within the project:

1. Enables semantic boundary detection for Semantic Chunking.
2. Generates vector representations of the final document chunks for storage in the vector database.

Using the same embedding model throughout the project ensures consistency during retrieval and evaluation.

---

## ?? Chunk Generation

Five independent chunking strategies were implemented to evaluate how different document partitioning methods affect retrieval performance and answer quality.

The implemented strategies include:

- Fixed Size Chunking
- Recursive Chunking
- Semantic Chunking
- Parent-Child Chunking
- Proposition Chunking

Each strategy produces a separate collection of document chunks, which are indexed independently to ensure a fair experimental comparison.

---

## ??? Vector Database Construction

After chunk generation, each collection of document chunks is converted into vector embeddings and stored in a dedicated vector database.

Instead of combining all chunking strategies into a single index, each strategy maintains its own vector database.

Independent vector databases provide:

- Isolated retrieval experiments
- Fair performance comparison
- Consistent evaluation settings
- Simplified retrieval analysis

---

## ?? Retriever Construction

Each vector database is transformed into a retriever responsible for identifying the most relevant document chunks for a given query.

The retriever performs semantic similarity search over stored embeddings and returns the most relevant contextual information to the language model.

Using identical retrieval settings across all experiments ensures that evaluation differences are attributed to the chunking strategy rather than retrieval configuration.

---

# ?? Chunking Strategies

Document chunking is a critical step in a Retrieval-Augmented Generation (RAG) pipeline. It determines how documents are divided before embedding generation and indexing in the vector database.

This project compares five chunking strategies using the same dataset, embedding model, retriever configuration, and evaluation metrics.

---

## 1. Fixed Size Chunking

Fixed Size Chunking divides documents into equal-sized chunks with a predefined chunk size and overlap.

**Characteristics**

- Simple and computationally efficient
- Easy to implement
- Produces uniform chunk sizes
- Suitable for structured documents

---

## 2. Recursive Chunking

Recursive Chunking splits documents hierarchically using separators such as paragraphs, sentences, and words while preserving document structure.

**Characteristics**

- Preserves document hierarchy
- Reduces unnecessary sentence fragmentation
- Maintains contextual boundaries

---

## 3. Semantic Chunking

Semantic Chunking uses an embedding model to identify semantically meaningful split points instead of relying solely on character or token counts.

**Characteristics**

- Context-aware chunk generation
- Groups semantically related information
- Suitable for complex technical documents

---

## 4. Parent-Child Chunking

Parent-Child Chunking indexes smaller child chunks while retrieving the corresponding parent chunk to provide richer contextual information.

**Characteristics**

- Fine-grained retrieval
- Better context preservation
- Improves contextual completeness

---

## 5. Proposition Chunking

Proposition Chunking divides documents into individual factual statements or propositions for highly granular retrieval.

**Characteristics**

- Fine-grained indexing
- Fact-oriented retrieval
- Precise context matching

---

All chunking strategies were evaluated independently using identical retrieval settings to ensure that performance differences resulted only from the chunking method.

---

# ?? Intermediate Artifacts (.pkl)

To improve development efficiency and reduce repeated computation, selected intermediate outputs are serialized and stored as `.pkl` (Pickle) files.

Instead of regenerating the same objects during every execution, the project loads serialized artifacts whenever available.

Examples include:

- Chunked document collections
- Parent–Child document mappings
- Processed retrieval objects
- Evaluation datasets (where applicable)

Benefits include:

- Faster experimentation
- Reduced execution time
- Reusable intermediate results
- Consistent evaluation across multiple runs

> **Note:** The generated `.pkl` files depend on the chunking strategy and execution workflow.

---

# ??? Vector Database & Retrieval

After document chunking, each chunk is converted into a vector embedding using the IBM Granite Embedding Model. These embeddings are indexed and stored in a vector database to enable semantic similarity search.

Each chunking strategy maintains an independent vector database to ensure a fair comparison.

The retrieval workflow is shown below:

User Query

?

Query Embedding

?

Similarity Search

?

Top-k Relevant Chunks

?

Context to LLM

?

Generated Response

The retriever performs semantic similarity search to identify the most relevant document chunks for a given query.

The retrieved context is then provided to the IBM Granite Large Language Model (LLM), which generates a context-aware response.

Maintaining separate vector databases and identical retrieval settings ensures that differences in performance are attributed to the chunking strategy rather than the retrieval infrastructure.

---

# ?? Hugging Face Dataset Creation

Before evaluation, the retrieved results and generated responses are organized into a Hugging Face `Dataset`. This standardized format allows seamless integration with the Ragas evaluation framework.

Each evaluation record contains:

- User Query
- Ground Truth Answer
- Retrieved Context
- Generated Response

The dataset is constructed after the response generation stage and serves as the input for automated RAG evaluation.

Using a standardized dataset format ensures consistent evaluation across all chunking strategies and simplifies the comparison of experimental results.

---

# ?? RAG Evaluation Methodology

The performance of each chunking strategy was evaluated using the **Ragas** evaluation framework.

To ensure a fair comparison, all strategies were tested using the same:

- Source documents
- Embedding model
- Large Language Model
- Retriever configuration
- Evaluation dataset

The following metrics were used:

| Metric | Description |
|---------|-------------|
| **Faithfulness** | Measures whether the generated response is supported by the retrieved context. |
| **Answer Relevancy** | Evaluates how well the generated response addresses the user's query. |
| **Context Precision** | Measures how much of the retrieved context is relevant to the query. |
| **Context Recall** | Measures whether the retriever successfully retrieved all the information required to answer the query. |

The evaluation process consisted of the following steps:

1. Retrieve relevant document chunks for each query.
2. Generate responses using the IBM Granite LLM.
3. Organize the evaluation data into a Hugging Face `Dataset`.
4. Compute evaluation metrics using the Ragas framework.
5. Compare the results across all chunking strategies.

Using a common evaluation pipeline ensures that the observed performance differences are primarily influenced by the chunking strategy rather than variations in the evaluation process.

---

# ?? Experimental Results

Each chunking strategy was evaluated using the same dataset, embedding model, language model, retriever configuration, and evaluation methodology. This ensured that the comparison remained consistent and that performance differences were primarily influenced by the chunking strategy.

## Evaluation Results

| Chunking Strategy | Faithfulness | Answer Relevancy | Context Precision | Context Recall |
|-------------------|-------------:|-----------------:|------------------:|---------------:|
| Semantic | 0.8519 | 0.6386 | 0.2778 | 0.4167 |
| Proposition | 0.6667 | 0.6121 | 0.3333 | 0.5833 |
| Parent-Child | **1.0000** | 0.6386 | 0.3333 | 0.4167 |
| **Fixed Size** | **1.0000** | **0.9976** | **0.7778** | **1.0000** |
| Recursive | 0.8333 | 0.6608 | 0.3333 | 0.4167 |

---

## Key Observations

- **Fixed Size Chunking** achieved the highest overall performance across the evaluated metrics.
- **Parent-Child Chunking** produced highly faithful responses but retrieved less relevant context than Fixed Size Chunking.
- **Semantic Chunking** generated reliable responses but showed lower retrieval performance for this dataset.
- **Recursive** and **Proposition Chunking** produced competitive results but did not outperform Fixed Size Chunking under the selected evaluation settings.

---

## Discussion

Although advanced chunking techniques are often recommended for production RAG systems, the experimental results obtained in this project indicate that **Fixed Size Chunking performed best for the IBM watsonx.ai documentation dataset**.

These findings should be interpreted within the context of this project. The effectiveness of a chunking strategy depends on several factors, including document structure, chunk size, embedding model, retrieval configuration, and evaluation dataset. Therefore, the results presented here should not be considered universally applicable to all RAG systems.

---

# ?? Challenges Faced & Lessons Learned

During the implementation and evaluation of the RAG pipeline, several practical challenges were encountered.

## Challenges Faced

- Configuring and authenticating the IBM watsonx.ai environment.
- Managing Python package dependencies and compatibility issues.
- Handling the computational cost of embedding generation and advanced chunking strategies.
- Creating and maintaining independent vector databases for each chunking strategy.
- Organizing evaluation data into the required Hugging Face `Dataset` format.
- Ensuring a fair comparison across all chunking strategies.

---

## Lessons Learned

Key takeaways from this project include:

- The effectiveness of a chunking strategy depends on the dataset and overall RAG pipeline.
- Simpler approaches can outperform more complex methods for well-structured technical documentation.
- A fair experimental setup is essential for meaningful comparisons.
- Modular implementation and reusable intermediate artifacts improve experimentation and reproducibility.
- Evaluation metrics provide valuable insights into retrieval quality and response generation.

---

# ?? Future Improvements

Potential enhancements for this project include:

- Hybrid Search (BM25 + Vector Search)
- Cross-Encoder or LLM-based reranking
- Evaluation using additional embedding models and LLMs
- Larger and more diverse datasets
- Adaptive chunk size selection
- Context Compression
- GraphRAG
- Agentic RAG
- Automated hyperparameter tuning
- Interactive web application for document querying

These improvements would further enhance retrieval quality, scalability, and production readiness.

---

# ?? References

The implementation and evaluation presented in this project were developed using the following resources:

- IBM watsonx.ai Documentation
- LangChain Documentation
- IBM Granite Model Documentation
- Hugging Face Datasets Documentation
- FAISS Documentation
- Chroma Documentation
- Ragas Documentation

---

# ????? Author

**Vamsi Kanugula**

**GitHub:** *(Add your GitHub Profile)*

**LinkedIn:** *(Add your LinkedIn Profile)*

If you found this project useful, consider giving the repository a ?.
