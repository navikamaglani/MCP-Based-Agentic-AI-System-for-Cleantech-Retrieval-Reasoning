#  MCP-Based Agentic AI System for Cleantech Retrieval & Reasoning

This repository contains a fully implemented **agentic AI system** built using the **Model Context Protocol (MCP)** to perform **document retrieval, reranking, summarization, and reasoning** over a large cleantech media corpus.  
Unlike polished academic demos, this project reflects the *real engineering journey* — including tool orchestration, ChromaDB indexing failures, debugging cycles, evaluation scripts, and all components that were actually built.

---

## Overview

The system operates across four main capabilities:

###  Retrieval  
- 20,111 cleantech media articles indexed in **ChromaDB**  
- Embeddings: `MiniLM-L6-v2`  
- Metadata stored: id, title, URL, date, author, domain  

###  Reranking  
- Cross-encoder: `ms-marco-MiniLM-L-6-v2`  
- Stronger relevance ordering than pure vector search  

###  Summarization  
- HuggingFace Inference API  
- Model: `meta-llama/Llama-3.2-3B-Instruct:free`  
- Grounded summarization using context windows  

###  Reasoning (Custom MCP Tool)  
Generates a justification for answers:  
- Supported vs. partially supported vs. weakly supported  
- References retrieved docs  
- Flags hallucinated or unsupported claims  

---

##  System Architecture

- **ChromaDB** stores 20k+ cleantech articles with embeddings  
- **MCP Server** exposes 4 tools:
  - `retrieve_cleantech`
  - `summarize_docs`
  - `explain_answer`
  - `answer_question`
- **Evaluation Pipeline** computes retrieval metrics + semantic answer similarity  
- **Offline scripts** reuse the same retrieval+reranking logic for large-scale benchmarking  

Diagrams for architecture and dataflow can be placed in the `diagrams/` folder.

---

##  Repository Structure

```
cleantech_mcp_agent/
│
├── mcp_server/
│   ├── server.py
│   └── __init__.py
│
├── src/
│   ├── build_index.py
│   ├── embedding_wrapper.py
│   ├── evaluate_system.py
│   ├── run_full_evaluation.py
│   ├── run_mcp_answers.py
│   │
│   ├── mcp_answers.csv
│   ├── results_retrieval.csv
│   ├── results_answers.csv
│   └── results_model_answers.json
│
├── data/
│   ├── cleantech_media_dataset_v3_2024-10-28.csv
│   ├── cleantech_rag_evaluation_data_2024-09-20.csv
│   └── CleanTech-50answers.txt
│
├── chroma_db/          
│
├── diagrams/           
│
├── requirements.txt
│
└── README.md
```


---

##  Installation

###  Clone the repo

###  Install dependencies

### (Optional but recommended)  
Ignore the vector DB directory to avoid corruption


Tools available:

| Tool | Purpose |
|------|---------|
| `ping` | Health check |
| `retrieve_cleantech` | Vector retrieval + metadata filters + reranking |
| `summarize_docs` | Context building + LLM summarization |
| `explain_answer` | Step-by-step reasoning + support level |
| `answer_question` | Full pipeline orchestrator |

---

##  Evaluation

All experiments follow the assignment’s Part 3 evaluation protocol.

### Run retrieval + answer quality benchmarks:


Outputs:

- `results_retrieval.csv`  
- `results_answers.csv`  
- `results_model_answers.json`  

### Retrieval Results (Top-10)

| Metric | Score |
|--------|--------|
| **Hit@10** | 0.34 |
| **Precision@10** | 0.068 |
| **Recall@10** | 0.513 |
| **MRR** | 0.260 |

**Interpretation:**  
The system retrieves **coverage well** (high recall), even if exact matches are noisy. This is expected in a domain with many overlapping articles.

### Answer Quality

| Metric | Score |
|--------|--------|
| **Mean Cosine Similarity (system vs. gold answers)** | **0.6586** |

This indicates the generated answers are *semantically aligned* with the gold references when retrieval succeeds.

---

##  Key Challenges Encountered

This project intentionally documents **real engineering hurdles**, including:

- ChromaDB schema mismatches & corruption  
- Batch-size limits during embedding  
- LangChain deprecations (HFEmbeddings → langchain-huggingface)  
- Strict JSON formatting in MCP stdio transport  
- HuggingFace inference failures / fallbacks  
- Environment inconsistencies  
- MCP tool invocation errors (`params.name`, wrong fields, etc.)

These shaped the final design and contributed to a more stable architecture.

---

##  Lessons Learned

### Technical  
- MCP is powerful but extremely strict  
- Reranking significantly improves retrieval quality  
- Version pinning is essential  
- Batching prevents Chroma from breaking  
- Reasoning tools expose hallucinations early  

### Project-Level  
- Real-world systems never work on the first try  
- Resetting & rebuilding indices is sometimes unavoidable  
- Keep scope tight to ensure the project ships  

---






