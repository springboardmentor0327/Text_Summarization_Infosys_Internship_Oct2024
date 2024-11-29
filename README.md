# Project Overview
This project delves into the field of **Text Summarization**, employing a variety of **Natural Language Processing (NLP)** techniques to condense lengthy documents into concise summaries while retaining their core information and context. The project explores **extractive**, **abstractive**, and **Large Language Model (LLM)**-based approaches, showcasing state-of-the-art methodologies.

---

# Project Objective
- **Implement and compare multiple summarization techniques**:
  - **Extractive Methods**: Select significant sentences from the text.
  - **Abstractive Methods**: Generate new summaries in natural language.
  - **LLM-Based Summarization**: Employ advanced models for intelligent summarization.
- **Evaluate each method** using standard metrics such as **ROUGE** and **BLEU** scores.

---

# Techniques Used

## Extractive Summarization
Directly selects key sentences from the original text. Algorithms implemented include:
- **Frequency-Based Summarization**: Highlights sentences with frequently occurring keywords.
- **Latent Semantic Analysis (LSA)**: Utilizes singular value decomposition for sentence extraction.
- **TextRank and LexRank**: Graph-based ranking techniques inspired by PageRank.
- **Luhn Algorithm**: Focuses on high keyword density for summarization.

## Abstractive Summarization
Generates concise summaries by rephrasing the content. Models used include:
- **T5 (Text-to-Text Transfer Transformer)**: Variants like `t5-small`, `t5-base`, `t5-large`.
- **BART (Bidirectional and Auto-Regressive Transformers)**: Variants like `bart-base`, `bart-large`.

## LLM-Based Summarization
Advanced techniques leveraging state-of-the-art models:
- **Basic Summarization**: Using APIs for quick results.
- **Map-Reduce**: Splits and summarizes text chunks iteratively.
- **Refinement**: Continuously improves summaries for coherence.

---

# Key Features
- **Multi-Algorithm Support**: Includes both extractive and abstractive methods.
- **Interactive Applications**: Integrates summarization into user-friendly apps using **Gradio** and **Streamlit**.
- **Evaluation Metrics**: Incorporates **ROUGE** and **BLEU** for quality assessment.
- **Scalability**: Handles large documents efficiently through chunking and aggregation techniques.
- **Deployment Ready**: Functional tools for summarization in real-world use cases such as journalism, education, and legal research.

---
