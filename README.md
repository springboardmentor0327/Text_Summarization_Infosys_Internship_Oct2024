# Text_Summarization_Infosys_Internship_Oct2024
This Text Summarization Project aims to develop a tool that efficiently condenses long articles, papers, or documents into concise summaries while preserving the key information and context. Utilizing advanced Natural Language Processing (NLP) techniques, this project focuses on both extractive and abstractive summarization methods.

## Project Objective

Develop Efficient Summarization Tools: Create tools to condense lengthy texts into concise summaries while retaining essential information.

Compare Summarization Techniques: Implement and assess the effectiveness of extractive, abstractive, and LLM-based methods using metrics like ROUGE and BLEU.

Enhance User Experience with Summarizing UI: Integrate summarization methods into a user-friendly Gradio interface for easy input of text or PDF files, method selection, and real-time summary generation.

Support Diverse Applications: Apply summarization tools to various domains, including news articles, legal documents, research papers, and customer support.

## Types of Text Summarization 
### Extractive Summarization  
  - Selects key sentences, phrases, or paragraphs directly from the source text without rephrasing. - Methods include   - Frequency-Based, LexRank, TextRank, and Luhn summarization. 
### Abstractive Summarization  
  - Generates summaries by understanding the text and writing new, concise versions in natural language. 
  - Methods include T5 (Text-to-Text Transfer Transformer) and BART (Bidirectional and Auto-Regressive Transformers). 
### Large Language Models (LLMs) 
  - Advanced techniques using models like Google Gemini.
  - Includes Map-Reduce, Iterative Refinement, and PDF Summarization.

## Techniques Used 

### Extractive Summarization 
1. **Frequency-Based Summarization**: Uses word frequency to identify the most important sentences.
2. **LexRank**: A graph-based approach that scores sentences based on their similarity and importance.
3. **TextRank**: Similar to LexRank but leverages Google's PageRank algorithm.
4. **Luhn**: Identifies sentences with high keyword density.

### Abstractive Summarization 
1. **T5**: Treats summarization as a text-to-text problem.
2. **BART**: Combines bidirectional encoding and auto-regressive decoding for high-quality summaries.

### Large Language Models (LLMs) 
1. **Map-Reduce**: Splits text into chunks, summarizes each, and combines them.
2. **Iterative Refinement**: Continuously improves the summary based on feedback and context.
3. **PDF Summarizer**: Summarizes content directly from PDF documents.

## Evaluation Metrics 
1. **ROUGE (Recall-Oriented Understudy for Gisting Evaluation)**: Compares the overlap between the generated summary and reference summary using N-grams, word sequences, and longest common subsequences.
2. **BLEU (Bilingual Evaluation Understudy)**: Measures similarity between the generated summary and reference summary by comparing word-level N-grams.

## Gradio Implementation 
A Gradio Web Interface is used to make these techniques accessible: 
  - Upload text or PDF files.
  - Select the summarization technique/algorithm.
  - Generate the summary.

## Use Cases 
  - **Summarizing News Articles**: Generate concise summaries for daily news consumption.
  - **Legal Document Analysis**: Quickly extract the essence of lengthy legal texts.
  - **Research Paper Summarization**: Save time by reviewing concise summaries of scientific papers.
  - **Customer Support**: Summarize chat logs to resolve queries

## Conclusion
This project demonstrates the capability of various summarization techniques to effectively condense large texts while maintaining key information. By integrating these methods into an easy-to-use Gradio interface, users can easily generate summaries for different types of documents, enhancing productivity and information accessibility.
