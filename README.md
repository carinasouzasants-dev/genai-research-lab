# 🧠 GenAI Research Lab

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:0B1020,45:4F46E5,100:00D4FF&height=230&section=header&text=GenAI%20Research%20Lab&fontSize=44&fontColor=FFFFFF&animation=fadeIn&fontAlignY=35&desc=Experimenting%20%7C%20Evaluating%20%7C%20Building%20%7C%20Applying&descAlignY=58&descSize=18" width="100%" />
</p>

<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=JetBrains+Mono&size=19&duration=2800&pause=700&color=00D4FF&center=true&vCenter=true&width=900&lines=Hands-on+Generative+AI+experimentation;LLMs+%7C+RAG+%7C+Embeddings+%7C+Agents;From+individual+experiments+to+an+applied+AI+system;AgroGenAI+%7C+Generative+AI+for+Agricultural+Intelligence" alt="Typing animation" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Google%20Gemini-GenAI-4285F4?style=for-the-badge&logo=google&logoColor=white" />
  <img src="https://img.shields.io/badge/Experiments-12-6D5DFB?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Status-Active-00C853?style=for-the-badge" />
</p>

<p align="center">
  <a href="https://github.com/carinasouzasants-dev/genai-research-lab">
    <img src="https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github&logoColor=white" />
  </a>
</p>

---

## 🔎 Overview

**GenAI Research Lab** is a hands-on laboratory for studying, experimenting with, and integrating Generative AI techniques using Python and Google Gemini.

The project was built progressively rather than starting with a large application.

Each experiment isolates an AI engineering concept, tests it in practice, records the result, and uses the knowledge gained to build the next layer.

The laboratory evolves from:

```text
Prompt Engineering
        ↓
Structured Outputs
        ↓
Model Comparison
        ↓
LLM Evaluation
        ↓
RAG
        ↓
Embeddings
        ↓
Semantic / Vector Search
        ↓
RAG + Embeddings
        ↓
Tool Calling
        ↓
AI Agent
        ↓
AI Automation
        ↓
AgroGenAI
```

The final experiment, **AgroGenAI**, integrates several of these concepts into an experimental Generative AI application for agricultural intelligence.

---

## 🎯 Project Goals

The main goals of this laboratory are to develop practical understanding of:

- Large Language Models
- Prompt Engineering
- Structured Outputs
- LLM Evaluation
- Retrieval-Augmented Generation
- Embeddings
- Semantic Search
- Vector Search
- Tool Calling
- AI Agents
- AI Automation
- Applied Generative AI
- Responsible AI behavior
- Human-in-the-loop workflows

The focus is not simply on obtaining an answer from an LLM.

The goal is to understand:

> **How can Generative AI systems be designed, tested, evaluated, integrated and applied to real-world problems?**

---

# 🧪 Experiment Roadmap

| # | Experiment | Main Concept | Status |
|:--:|---|---|:--:|
| 01 | Prompt Comparison | Zero-shot vs Few-shot | ✅ |
| 02 | Structured Output | Schema-based generation | ✅ |
| 03 | Model Comparison | Gemini model comparison | ✅ |
| 04 | LLM Evaluation | Rubric-based evaluation | ✅ |
| 05 | Basic RAG | Knowledge retrieval | ✅ |
| 06 | Embeddings | Semantic representations | ✅ |
| 07 | Vector Search | Similarity search | ✅ |
| 08 | RAG + Embeddings | Semantic retrieval | ✅ |
| 09 | Tool Calling | LLM + external tools | ✅ |
| 10 | AI Agent | Reasoning + actions | ✅ |
| 11 | AI Automation | AI workflow automation | ✅ |
| 12 | AgroGenAI | Applied GenAI system | ✅ |

---

# 🧬 Technical Evolution

## 01 → Prompt Comparison

The first experiment investigates how different prompting strategies influence LLM responses.

### Compared

- Zero-shot prompting
- Few-shot prompting
- Response quality
- Practical usefulness
- Execution time

### Why it matters

Prompt design is one of the first control layers of an LLM application.

---

## 02 → Structured Output

The model is constrained to produce structured data instead of unrestricted text.

### Technologies

- Python
- Pydantic
- JSON
- Gemini Structured Outputs

### Goal

Make LLM responses easier to consume programmatically.

```text
LLM
 ↓
Schema
 ↓
Validated JSON
 ↓
Application
```

---

## 03 → Model Comparison

Different Gemini models are evaluated against the same task.

The experiment explores differences in:

- Response time
- Output size
- Detail
- Practical usefulness
- Overall response quality

The goal is to understand that **model selection is an engineering decision**, not simply a matter of choosing the newest model.

---

## 04 → LLM Evaluation

A simple evaluation framework was implemented to compare generated responses using a defined rubric.

### Evaluation dimensions

- Clarity
- Completeness
- Instruction adherence
- Quality
- Applicability
- Responsible use of hypotheses

This experiment introduces a fundamental AI engineering principle:

> **A generated response should be evaluated, not automatically trusted.**

---

## 05 → Basic RAG

The project introduces Retrieval-Augmented Generation using a local agricultural knowledge base.

Instead of relying only on the model's internal knowledge:

```text
User Query
    ↓
Knowledge Base
    ↓
Relevant Context
    ↓
LLM
    ↓
Response
```

The experiment demonstrates the basic relationship between retrieval and generation.

---

## 06 → Embeddings

Text is transformed into numerical vector representations using an embedding model.

### Embedding model

`gemini-embedding-2`

Embeddings allow text to be compared according to semantic similarity rather than only exact keyword matching.

```text
Text
 ↓
Embedding Model
 ↓
Vector Representation
 ↓
Semantic Comparison
```

---

## 07 → Vector Search

A local vector search mechanism was implemented using numerical similarity.

### Concepts explored

- Embeddings
- Vector representations
- Cosine similarity
- Similarity ranking
- Top-K retrieval

The experiment transforms the embedding concept into an actual retrieval mechanism.

---

## 08 → RAG + Embeddings

The previous experiments are combined into a semantic retrieval pipeline.

```text
Query
 ↓
Query Embedding
 ↓
Vector Similarity
 ↓
Top-K Documents
 ↓
Context
 ↓
LLM
 ↓
Generated Response
```

This represents a more realistic foundation for knowledge-grounded AI applications.

---

## 09 → Tool Calling

The LLM is connected to simulated external tools.

The experiment demonstrates how a model can determine when additional information should be retrieved through a function.

### Simulated tools

- Crop productivity
- Weather information

```text
User Request
      ↓
     LLM
      ↓
Tool Decision
      ↓
Function Call
      ↓
Tool Result
      ↓
     LLM
      ↓
Final Response
```

This introduces the connection between LLMs and external systems.

---

## 10 → AI Agent

Tool Calling is extended into an agentic workflow.

The agent can:

1. Analyze the request
2. Determine which tool is relevant
3. Execute the tool
4. Receive the result
5. Continue processing
6. Produce a final analysis

```text
              ┌───────────────┐
              │     User      │
              └───────┬───────┘
                      ↓
              ┌───────────────┐
              │   AI Agent    │
              └───────┬───────┘
                      ↓
               Choose Action
                      ↓
              ┌───────────────┐
              │     Tool      │
              └───────┬───────┘
                      ↓
                 Tool Result
                      ↓
                 Agent Loop
                      ↓
              Final Response
```

The experiment used multiple tool calls in a single reasoning workflow.

---

## 11 → AI Automation

The project then explores a more operational workflow.

An unstructured agricultural report is transformed into structured information that can be consumed by other systems.

```text
Agricultural Report
        ↓
       LLM
        ↓
Classification
        ↓
Structured Data
        ↓
   ┌────┴────┐
   ↓         ↓
 JSON       CSV
```

The generated structure includes:

- Category
- Priority
- Summary
- Identified factors
- Recommended action
- Human analysis requirement

The workflow also explicitly preserves the need for human validation when the available information is insufficient.

---

# 🌱 12 → AgroGenAI

## Generative AI for Agricultural Intelligence

**AgroGenAI** is the final experimental application of the laboratory.

It combines the concepts investigated throughout the previous experiments into a single pipeline.

The prototype explores how Generative AI can assist in analyzing agricultural information while maintaining:

- Context retrieval
- Semantic similarity
- Structured generation
- Explicit uncertainty
- Human validation

---

## 🏗️ AgroGenAI Architecture

```text
┌─────────────────────────────┐
│   Agricultural Report       │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│     Query Embedding         │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│      Semantic Search        │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  Relevant Knowledge Chunks  │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│        Gemini LLM           │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│     Structured Analysis     │
└──────────────┬──────────────┘
               │
       ┌───────┴────────┐
       ▼                ▼
 Recommendations   Risks & Limits
       │                │
       └───────┬────────┘
               ▼
     Human Validation
```

---

## 📥 Example Input

The prototype receives an experimental report describing a soybean production scenario involving:

- Reduced estimated productivity
- Irregular rainfall
- Possible water deficit
- Higher temperatures
- Possible soil changes
- Insufficient data to establish a definitive cause

---

## 📤 Example Output

The system produces a structured analysis containing:

### Category

`Agricultural Productivity / Climatic and Soil Factors`

### Priority

`Medium`

### Identified factors

- Reduced estimated productivity
- Irregular rainfall
- Possible water deficit
- Higher temperatures
- Possible soil changes

### Recommendations

The system suggests additional investigation using information such as:

- Historical productivity
- Meteorological records
- Soil characteristics
- Soil moisture
- Crop management
- Pest and disease information

### Human validation

```text
Requer análise humana: True
```

This is intentional.

The prototype is designed to **support analysis rather than present AI-generated hypotheses as definitive agricultural diagnoses**.

---

# 🧠 Responsible AI Approach

One of the principles of this laboratory is that an AI-generated answer should not automatically be treated as truth.

The experiments therefore emphasize:

### Evidence

Use available information to ground the response.

### Uncertainty

Distinguish observations from hypotheses.

### Limitations

Explicitly communicate when available data is insufficient.

### Human-in-the-loop

Keep domain specialists involved when decisions require expertise or additional validation.

This becomes particularly important in domains where incorrect conclusions can influence real-world decisions.

---

# ⚙️ Technology Stack

## Core

<p>
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Google%20Gemini-4285F4?style=flat-square&logo=google&logoColor=white" />
  <img src="https://img.shields.io/badge/Pydantic-E92063?style=flat-square&logo=pydantic&logoColor=white" />
  <img src="https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white" />
</p>

## Generative AI

```text
Large Language Models
Prompt Engineering
Structured Outputs
Embeddings
RAG
Semantic Search
Vector Search
Tool Calling
AI Agents
AI Automation
LLM Evaluation
```

## Development

```text
Python
Git
GitHub
Virtual Environments
Environment Variables
JSON
CSV
```

---

# 📦 Main Dependencies

```text
google-genai
python-dotenv
pydantic
numpy
```

The dependency list is intentionally kept focused on the components required by the laboratory.

---

# 📁 Repository Structure

```text
genai-research-lab/
│
├── data/
│   └── agricultural_knowledge.txt
│
├── experiments/
│   ├── __init__.py
│   ├── prompt_comparison_01.py
│   ├── structured_output_02.py
│   ├── model_comparison_03.py
│   ├── evaluation_04.py
│   ├── rag_05.py
│   ├── embeddings_06.py
│   ├── vector_search_07.py
│   ├── rag_embeddings_08.py
│   ├── tool_calling_09.py
│   ├── ai_agent_10.py
│   ├── ai_automation_11.py
│   └── agrogenai_12.py
│
├── results/
│   ├── structured_output_02.json
│   ├── model_comparison_03.json
│   ├── evaluation_04.json
│   ├── rag_05.json
│   ├── embeddings_06.json
│   ├── vector_search_07.json
│   ├── rag_embeddings_08.json
│   ├── tool_calling_09.json
│   ├── ai_agent_10.json
│   ├── ai_automation_11.json
│   ├── ai_automation_11.csv
│   └── agrogenai_12.json
│
├── src/
│   ├── __init__.py
│   └── gemini_client.py
│
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

---

# 🚀 Getting Started

## 1. Clone the repository

```bash
git clone https://github.com/carinasouzasants-dev/genai-research-lab.git
cd genai-research-lab
```

## 2. Create a virtual environment

```bash
python -m venv .venv
```

### macOS / Linux

```bash
source .venv/bin/activate
```

### Windows

```bash
.venv\Scripts\activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 4. Configure the API key

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_api_key_here
```

The `.env` file is excluded from version control.

Never commit API keys or other secrets.

---

# ▶️ Running an Experiment

Each experiment can be executed independently.

Example:

```bash
python experiments/prompt_comparison_01.py
```

Another example:

```bash
python experiments/ai_agent_10.py
```

To run the final AgroGenAI prototype:

```bash
python experiments/agrogenai_12.py
```

The AgroGenAI result is saved to:

```text
results/agrogenai_12.json
```

---

# 📊 Experimental Evidence

The laboratory stores structured outputs generated during experiments.

This makes it possible to inspect results after execution instead of relying only on terminal output.

Examples include:

```text
results/
├── model_comparison_03.json
├── evaluation_04.json
├── embeddings_06.json
├── vector_search_07.json
├── rag_embeddings_08.json
├── tool_calling_09.json
├── ai_agent_10.json
├── ai_automation_11.json
├── ai_automation_11.csv
└── agrogenai_12.json
```

The results directory therefore acts as an experimental record of the laboratory's progression.

---

# 🔍 Key Engineering Lessons

## 1. Prompting is an engineering variable

Small changes in instructions can influence model behavior and output quality.

## 2. Structured outputs make LLMs easier to integrate

Schemas turn free-form generation into data that software can validate and consume.

## 3. Retrieval can provide contextual grounding

RAG allows an application to provide relevant external information to the model.

## 4. Embeddings enable semantic retrieval

Meaning can be represented numerically and compared through vector similarity.

## 5. LLMs become more useful when connected to tools

Tool Calling allows models to interact with external functions and information.

## 6. Agents combine reasoning and actions

An agentic workflow can decide which action to perform and use its result in subsequent steps.

## 7. Automation transforms AI capabilities into workflows

The value of an LLM increases when its output can become structured data consumed by another process.

## 8. Evaluation is essential

A response being generated successfully does not mean that it is correct, useful or appropriate.

## 9. Human validation remains important

Especially when data is incomplete or the domain requires specialized expertise.

---

# 🧭 Design Philosophy

The laboratory follows a progressive engineering approach:

```text
Understand
    ↓
Experiment
    ↓
Measure
    ↓
Evaluate
    ↓
Integrate
    ↓
Apply
```

Instead of jumping directly into a complex AI application, each layer is investigated independently before being incorporated into a larger system.

This makes the evolution of the project explicit and reproducible.

---

# 🔐 Limitations

This repository is an experimental research and learning project.

The current implementation has several limitations:

- Agricultural information is experimental and limited.
- Some external data sources are simulated.
- The vector search implementation is local.
- The knowledge base is intentionally small.
- The evaluation methodology is still basic.
- AgroGenAI is not a production agricultural decision-support system.
- Generated recommendations require domain validation.
- Real-world deployment would require stronger evaluation, observability, security, data governance and reliability controls.

These limitations are part of the project's current scope and provide opportunities for future development.

---

# 🛣️ Future Roadmap

The next development stages may include:

- [ ] Streamlit interface for AgroGenAI
- [ ] Larger agricultural knowledge base
- [ ] Real agricultural datasets
- [ ] Persistent vector database
- [ ] Improved retrieval evaluation
- [ ] Retrieval quality metrics
- [ ] Automated experiment evaluation
- [ ] Real external API integrations
- [ ] Agent observability
- [ ] Token and cost tracking
- [ ] Latency benchmarking
- [ ] Automated tests
- [ ] Experiment dashboard
- [ ] Production-oriented architecture

---

# 💡 Why AgroGenAI?

Agriculture generates large amounts of information across different sources such as:

```text
Weather
Soil
Productivity
Crop Management
Historical Records
Operational Data
Pest & Disease Information
```

One potential role for Generative AI is to help users interact with these sources through natural language while connecting retrieved evidence to structured analysis.

AgroGenAI is an experimental exploration of this idea.

The objective is not to replace agricultural expertise.

The objective is to investigate how an AI system could help **organize information, retrieve relevant context, identify factors and support human analysis**.

---

# 👩🏻‍💻 About the Author

## Carina Souza

Software Engineering student focused on **Artificial Intelligence, Generative AI and AI Engineering**.

This laboratory represents a practical learning journey through the engineering of Generative AI systems, from individual LLM experiments to an integrated applied AI prototype.

### Current areas of interest

```text
Artificial Intelligence
Generative AI
Large Language Models
RAG
AI Agents
Python
Data
Automation
Software Engineering
Applied AI
```

---

# 🔗 Connect

<p align="center">
  <a href="https://github.com/carinasouzasants-dev">
    <img src="https://img.shields.io/badge/GitHub-Carina%20Souza-181717?style=for-the-badge&logo=github&logoColor=white" />
  </a>

  <a href="https://www.linkedin.com/in/carina-souza-dev">
    <img src="https://img.shields.io/badge/LinkedIn-Carina%20Souza-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white" />
  </a>
</p>

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:00D4FF,50:4F46E5,100:0B1020&height=130&section=footer" width="100%" />
</p>

<p align="center">
  <strong>Experiment. Evaluate. Build. Apply. 🚀</strong>
</p>
