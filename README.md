# Text_Summarization_Infosys_Internship_Oct2024
This Text Summarization Project aims to develop a tool that efficiently condenses long articles, papers, or documents into concise summaries while preserving the key information and context. Utilizing advanced Natural Language Processing (NLP) techniques.
It covers both extractive and abstractive approaches, as well as advanced techniques using large language models (LLMs). The application includes ROUGE and BLEU score evaluation for summarization quality and a Gradio-based user interface that accepts input as text or PDFs.

**Project Overview**
Text summarization aims to create concise, meaningful summaries of input text or documents. This project provides a comprehensive solution with:
**1**.Extractive Summarization: Identifies and extracts the most relevant sentences from the input.
**2**.Abstractive Summarization: Generates summaries by paraphrasing and reinterpreting the input.
**3**.Advanced Summarization Techniques: Uses LLMs with refined approaches like MapReduce and Refine for scalable and context-aware summarization.

**Techniques and Models**
**1.Extractive Summarization**
  **->Frequency Method**: Summarizes text based on word frequencies.
  **->Sumy Method**: Utilizes the Sumy library for extracting the most relevant sentences.
  **->Luhn Method**: Identifies important sentences using word frequencies and positions in the text.

**2. Abstractive Summarization**
  **->T5 (Text-to-Text Transfer Transformer)**: A transformer model fine-tuned for summarization tasks.
  **->BART (Bidirectional and Auto-Regressive Transformers)**:Generates high-quality abstractive summaries by reconstructing     corrupted text inputs.


**3. Advanced Summarization**
  **->Refine Method**:Incrementally refines a generated summary by considering user-defined prompts or intermediate outputs.
  **->MapReduce Method**:Splits large documents into smaller parts, summarizes them, and combines the results for scalability    and efficiency.

**Evaluation Metrics**
To assess the quality of summaries:
**->ROUGE (Recall-Oriented Understudy for Gisting Evaluation)**:Measures overlap of n-grams, word sequences, and word pairs between the system-generated summary and the reference.
**->BLEU (Bilingual Evaluation Understudy)**:Evaluates the fluency and relevance of the generated summary by comparing it with the reference.

**Gradio Application**
A user-friendly interface is built using Gradio, allowing users to interact with the summarization models.
**Key Features:**
**1.Input Options**:
  **->Text Input**:Enter plain text for summarization.
  **->PDF Input**:Upload a PDF, and the application extracts text and summarizes it.
**2.Model Selection:**
  **->**For extractive methods: Choose **Frequency, Sumy**, or **Luhn**.
  **->**For abstractive methods: Choose **T5** or **BART**.
  **->**For advanced methods: Utilize **LLMs with Refine or MapReduce**.
**Customizable Parameters**
  **->**Set the **maximum word limit** for summaries.
 **->**View word counts and summary outputs dynamically.
**4.Outputs**
  **->**Summarized text.
  **->**Word count of the summary.

  
**Implementation**
**->Technologies Used**
  **1.Python**: Core language for implementation.
  **2.Libraries:**
    **NLTK**: Natural Language Toolkit for text processing.
    **SpaCy**: NLP library for efficient text parsing.
    **Sumy**: Library for extractive summarization.
    **Transformers (Hugging Face)**: Framework for implementing T5, BART, and LLMs.
    **PyPDF2**: For PDF text extraction.
    **Gradio**: For building the user interface.

**Example Use Cases**
  **1.Academic Research**: Summarize lengthy research papers into concise abstracts.
  **2.Business**: Generate executive summaries from reports and proposals.
  **3.Education**: Simplify textbook content for students.

**Future Enhancements**
  **1.**Integration with cloud storage for document uploads (e.g., Google Drive, Dropbox).
  **2.**Fine-tuning models for domain-specific summaries.
  **3.**Real-time multi-lingual summarization support.
