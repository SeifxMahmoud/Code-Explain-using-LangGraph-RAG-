# 🧠 Code-Explain with LangGraph & Retrieval-Augmented Generation (RAG)

This repository contains an implementation of a **code assistant** that can both **generate** new code and **explain existing code**. The system is built on top of [LangGraph](https://python.langchain.com/docs/langgraph) for state-driven orchestration, combined with **Retrieval-Augmented Generation (RAG)** to improve few-shot prompting.  

It integrates:
- **Semantic intent classification** (deciding whether to *generate* or *explain*).
- **RAG-based context retrieval** using FAISS and SentenceTransformers.
- **LLM-driven code generation** with Salesforce CodeGen.
- **AST-based code explanation** for structured breakdowns of user-provided code.
- A **Gradio UI** for interactive usage.

---

## 🔍 Overview

The project explores:

- Using **SentenceTransformers + FAISS** for semantic retrieval of similar HumanEval tasks.  
- Constructing few-shot **generation prompts** enriched with retrieved context.  
- Employing **CodeGen-350M-mono** to synthesize Python code.  
- Implementing **AST parsing** to explain code logic step by step.  
- Leveraging **LangGraph** to orchestrate states: intent classification → routing → code generation or explanation.  
- Deploying a **Gradio app** for an interactive user interface.  

This work is a practical attempt at integrating **RAG workflows**, **LLM-powered code generation**, and **explainability pipelines**.

---

## 📁 Directory Structure

```bash
├── data/               # HumanEval dataset for retrieval context
├── LangGraph.py        # Core project code (pipeline, graph, UI, tests)
├── requirements.txt    # Dependencies
└── README.md           # Project documentation
