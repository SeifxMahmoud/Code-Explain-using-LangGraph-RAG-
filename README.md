🧠 Code-Explain with LangGraph & Retrieval-Augmented Generation (RAG)

This repository contains an implementation of a code assistant that can both generate new code and explain existing code. The system is built on top of LangGraph
 for state-driven orchestration, combined with Retrieval-Augmented Generation (RAG) to improve few-shot prompting.

It integrates:

Semantic intent classification (deciding whether to generate or explain).

RAG-based context retrieval using FAISS and SentenceTransformers.

LLM-driven code generation with Salesforce CodeGen.

AST-based code explanation for structured breakdowns of user-provided code.

A Gradio UI for interactive usage.

🔍 Overview

The project explores:

Using SentenceTransformers + FAISS for semantic retrieval of similar HumanEval tasks.

Constructing few-shot generation prompts enriched with retrieved context.

Employing CodeGen-350M-mono to synthesize Python code.

Implementing AST parsing to explain code logic step by step.

Leveraging LangGraph to orchestrate states: intent classification → routing → code generation or explanation.

Deploying a Gradio app for an interactive user interface.

This work is a practical attempt at integrating RAG workflows, LLM-powered code generation, and explainability pipelines.

📁 Directory Structure
├── data/               # HumanEval dataset for retrieval context
├── LangGraph.py        # Core project code (pipeline, graph, UI, tests)
├── requirements.txt    # Dependencies
└── README.md           # Project documentation

⚙️ Setup

Clone the repository

git clone https://github.com/yourusername/Code-Explain-LangGraph-RAG.git
cd Code-Explain-LangGraph-RAG


Create a virtual environment

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate


Install dependencies

pip install -r requirements.txt


(Optional) Upgrade accelerate for optimized inference

pip install --upgrade accelerate

🚀 Run the App

Launch the Gradio interface:

python LangGraph.py


Then open the local URL shown in the terminal.

You can try inputs like:

"Generate a function that sorts a list"

"Explain this code: for i in range(5): print(i)"

✅ Tests

The project includes some quick validation tests inside LangGraph.py:

Function generation (e.g., add two numbers, sorting a list).

Code explanation (loops, functions).

Invalid input handling (non-code queries).

Run:

python LangGraph.py


and check the console for test outputs.
