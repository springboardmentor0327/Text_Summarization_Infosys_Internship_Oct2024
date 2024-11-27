# Text_Summarization_Infosys_Internship_Oct2024
This Text Summarization Project aims to develop a tool that efficiently condenses long articles, papers, or documents into concise summaries while preserving the key information and context. Utilizing advanced Natural Language Processing (NLP) techniques, this project focuses on both extractive and abstractive summarization methods.

Welcome to the **Text Summarization Project**, a comprehensive exploration of techniques to condense lengthy articles, papers, and documents into concise summaries while preserving the original context and essential information. The project leverages advanced **Natural Language Processing (NLP)** methods and state-of-the-art models, including **extractive**, **abstractive**, and **Large Language Models (LLMs)** approaches.

---

## **Project Objective**
The goal of this project is to create a tool capable of efficiently summarizing textual content through a combination of methods, including:
- **Extractive Summarization**: Identifying and extracting key sentences from the text.
- **Abstractive Summarization**: Generating summaries by paraphrasing the content.
- **LLM-Based Summarization**: Utilizing Large Language Models for advanced summarization techniques.

---

## **Folder Structure**
The project is organized into the following directories, corresponding to different stages of development and experimentation:

### **1. Week1: Extractive Summarization**
- **File**: `extractive_all3.ipynb`
- **Description**: Implements three extractive summarization algorithms:
  - **TextRank**
  - **Latent Semantic Analysis (LSA)**
  - **Frequency-Based Summarization**

---

### **2. Week2: Abstractive Summarization**
- **Files**:
  - `abstractive_bart.ipynb`: Implements **BART**-based summarization with variants:
    - `bart-large-cnn`
    - `bart-large`
    - `bart-base`
  - `abstractive_t5.ipynb`: Implements **T5**-based summarization with variants:
    - `t5-small`
    - `t5-large`
    - `t5-base`

---

### **3. Week3-4: Enhanced Models and Evaluation**
- **Files**:
  - `abstractive_LLM.ipynb`: Demonstrates abstractive summarization using **Gemini-1.5-Flash LLM**.
  - Other files in this folder are Modified versions of extractive and abstractive notebooks (from Week1 and Week2) to include:
    - **ROUGE** score calculation.
    - **2-gram BLEU** score evaluation.

---

### **4. Week5-6: Advanced Techniques**
- **File**: `refine&map_reduce.ipynb`
- **Description**: Implements advanced summarization techniques:
  - **Refine**: Iteratively improves summaries using context refinement.
  - **Map-Reduce**: Splits input into smaller chunks for summarization and combines results.

---

### **5. Week7-8: Demonstration and Deployment**
- **Files**:
  - `summary_app.ipynb`: A fully functional **Streamlit-based web app** that integrates all summarization methods, allowing users to interact with the tool via a graphical interface.
  - `demo_video.mp4`: A walkthrough video demonstrating the functionality and usage of the summarization app.

---

## **Features**
- **Multi-Algorithm Support**: Includes both extractive and abstractive methods, covering a wide range of summarization techniques.
- **Evaluation Metrics**: Integrated ROUGE and BLEU scoring for performance evaluation.
- **Large Language Models**: Leverages state-of-the-art models like **Gemini-1.5-Flash**.
- **Interactive App**: A user-friendly web app for summarization tasks.
- **Scalability**: Handles large documents by splitting content into manageable chunks using the **Map-Reduce** approach.

---

## **Setup Instructions**
Follow these steps to set up the project locally:

### **1. Clone the Repository**
```bash
git clone https://github.com/your-username/text-summarization-project.git
cd text-summarization-project
```

### **2. Install Dependencies**
Install the required Python packages using:
```bash
pip install -r requirements.txt
```

### **3. Run the Streamlit App**
Navigate to the `week7-8` folder and execute:
```bash
streamlit run summary_app.ipynb
```

---

## **Usage**
### **1. Jupyter Notebooks**
- Explore the notebooks in each week’s folder to understand the summarization techniques and algorithms.
- Experiment with different configurations of extractive and abstractive models.

### **2. Web App**
- Upload text or PDF files and generate summaries interactively.
- Compare extractive, abstractive, and LLM-based methods.

---

## **Technologies and Libraries**
- **NLP Libraries**: `Transformers`, `Sumy`, `nltk`, `langchain`
- **Models**: `BART`, `T5`, `Gemini-1.5-Flash`
- **Metrics**: `ROUGE`, `BLEU`
- **Frameworks**: `Streamlit`
- **File Handling**: `PyPDF2`, `FPDF`, `python-docx`

---



