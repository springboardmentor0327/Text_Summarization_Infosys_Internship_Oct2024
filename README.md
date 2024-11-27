# Text_Summarization_Infosys_Internship_Oct2024
This Text Summarization Project aims to develop a tool that efficiently condenses long articles, papers, or documents into concise summaries while preserving the key information and context. Utilizing advanced Natural Language Processing (NLP) techniques, this project focuses on both extractive and abstractive summarization methods.

---

## **Types of Text Summarization**

1. **Extractive Summarization**:
   - Selects key sentences, phrases, or paragraphs directly from the source text.
   - Does not rephrase or modify the original content.

2. **Abstractive Summarization**:
   - Generates a summary by understanding the meaning of the text and writing a new, concise version in natural language.
   - Involves rephrasing and interpreting the source.
   - Uses advanced LLM models like Google Gemini to summarize text by understanding context and generating coherent and concise outputs.

---

## **Techniques Used**

### 1. **Extractive Summarization**
We implemented the following algorithms for extractive summarization:
- **Frequency-Based Summarization**:
  - Uses word frequency to identify the most important sentences.
- **LexRank**:
  - A graph-based approach that scores sentences based on their similarity and importance.
- **TextRank**:
  - Similar to LexRank but leverages Google's PageRank algorithm to rank sentences.

### 2. **Abstractive Summarization**
Abstractive summarization methods involve the following state-of-the-art models:
- **T5 (Text-to-Text Transfer Transformer)**:
  - A versatile model that treats summarization as a text-to-text problem.
- **BART (Bidirectional and Auto-Regressive Transformer)**:
  - Combines bidirectional encoding and auto-regressive decoding for high-quality summaries.

### 3. **Large Language Models (LLMs)**
Used **Google Gemini** to implement advanced summarization techniques:
- **Map-Reduce**:
  - Splits text into chunks, summarizes each, and combines them for a cohesive result.
- **Iterative Refinement**:
  - Iteratively improves the quality of the summary based on feedback and context.
- **PDF Summarizer**:
  - Summarizes content directly from PDF documents, enabling seamless document processing.

---

## **Evaluation Metrics**  
To evaluate the quality of generated summaries, we use the following metrics:  

### **1. ROUGE (Recall-Oriented Understudy for Gisting Evaluation)**  
- Compares the overlap between the generated summary and reference summary using N-grams, word sequences, and longest common subsequences.  

### **2. BLEU (Bilingual Evaluation Understudy)**  
- Measures how similar the generated summary is to the reference summary by comparing word-level N-grams.  
- Typically used for machine translation but applied here for summarization.  

---

## **Gradio Implementation**
To make these techniques accessible, we integrated them into a **Gradio Web Interface**. Users can:
- Upload text or PDF files.
- Select the summarization technique/algorithm.
- Generate the summary.
