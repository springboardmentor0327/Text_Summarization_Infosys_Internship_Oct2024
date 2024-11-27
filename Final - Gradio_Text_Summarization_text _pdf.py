
from PyPDF2 import PdfReader
import os
from dotenv import load_dotenv, dotenv_values 

load_dotenv() 

# Define your summarization functions
def extractive_summarization_frequency(txt, target_word_count, min_sentence_word_count=10):
    import nltk
    import heapq
    from nltk.corpus import stopwords
    from nltk.tokenize import sent_tokenize, word_tokenize

    nltk.download('punkt')
    nltk.download('stopwords')

    def summarize_text(text, target_word_count, min_sentence_word_count):
        # Tokenize sentences
        sentences = sent_tokenize(text)

        # Preprocess text to filter out non-alphabetic words and stopwords
        def preprocess_text(text):
            processed_words = []
            for word in word_tokenize(text):
                if word.isalpha():
                    processed_words.append(word.lower())
            return processed_words

        words = preprocess_text(text)

        stop_words = set(stopwords.words('english'))
        filtered_words = [word for word in words if word not in stop_words]

        # Calculate word frequencies
        word_frequencies = {}
        for word in filtered_words:
            if word in word_frequencies:
                word_frequencies[word] += 1
            else:
                word_frequencies[word] = 1

        # Normalize word frequencies
        max_frequency = max(word_frequencies.values())
        for word in word_frequencies:
            word_frequencies[word] /= max_frequency

        # Score sentences based on word frequencies
        sentence_scores = {}
        for sentence in sentences:
            sentence_words = preprocess_text(sentence)
            for word in sentence_words:
                if word in word_frequencies:
                    if len(sentence.split(' ')) >= min_sentence_word_count:
                        if sentence in sentence_scores:
                            sentence_scores[sentence] += word_frequencies[word]
                        else:
                            sentence_scores[sentence] = word_frequencies[word]

        # Sort sentences by score
        sorted_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)

        # Select sentences until target word count is met, ensuring min_sentence_word_count
        summary_sentences = []
        word_count = 0
        for sentence in sorted_sentences:
            sentence_word_count = len(word_tokenize(sentence))
            if word_count + sentence_word_count <= target_word_count:
                summary_sentences.append(sentence)
                word_count += sentence_word_count
            elif not summary_sentences and sentence_word_count >= min_sentence_word_count:
                # If no sentences have been added and the least word-count sentence meets min_sentence_word_count, add it
                summary_sentences.append(sentence)
                break
            else:
                break  # Stop if we exceed the target word count

        return " ".join(summary_sentences)  # Return summary as a string

    # Replace any user-specified "\n" with actual line breaks
    text = txt.replace("\\n", "\n")

    # Summarize the text with the target word count
    return summarize_text(text, target_word_count, min_sentence_word_count)


def extractive_summarization_tfidf(txt, target_word_count, min_sentence_word_count=10):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from nltk.tokenize import sent_tokenize
    import numpy as np
    import nltk
    nltk.download('punkt')

    def summarize_text(text, target_word_count, min_sentence_word_count):
        # Tokenize sentences
        sentences = sent_tokenize(text)

        # Generate the TF-IDF matrix
        tfidf = TfidfVectorizer()
        tfidf_matrix = tfidf.fit_transform(sentences)

        # Calculate sentence scores by summing TF-IDF values for each sentence
        sentence_scores = np.sum(tfidf_matrix.toarray(), axis=1)

        # Get indices of sentences sorted by score
        sorted_sentence_indices = np.argsort(sentence_scores)[::-1]

        # Select sentences until target word count is met, respecting min_sentence_word_count
        summary_sentences = []
        word_count = 0
        for i in sorted_sentence_indices:
            sentence = sentences[i]
            sentence_word_count = len(sentence.split())
            if word_count + sentence_word_count <= target_word_count:
                summary_sentences.append(sentence)
                word_count += sentence_word_count
            elif not summary_sentences and sentence_word_count >= min_sentence_word_count:
                # If no sentences have been added and this sentence meets the minimum word count, add it
                summary_sentences.append(sentence)
                break
            else:
                break  # Stop if we exceed the target word count

        return " ".join(summary_sentences)  # Return summary as a string

    # Replace any user-specified "\n" with actual line breaks
    text = txt.replace("\\n", "\n")

    # Summarize the text with the target word count
    return summarize_text(text, target_word_count, min_sentence_word_count)


def extractive_summarization_lsa(txt, target_word_count, min_sentence_word_count=10):
    from sumy.parsers.plaintext import PlaintextParser
    from sumy.nlp.tokenizers import Tokenizer
    from sumy.summarizers.lsa import LsaSummarizer

    def summarize_text(text, target_word_count, min_sentence_word_count):
        # Parse the text
        parser = PlaintextParser.from_string(text, Tokenizer("english"))

        # Initialize LSA summarizer
        summarizer = LsaSummarizer()

        # Generate initial summary (retrieve all sentences scored by LSA)
        summary = summarizer(parser.document, len(parser.document.sentences))

        # Sort sentences by relevance and accumulate until reaching the target word count
        summary_sentences = []
        word_count = 0
        for sentence in summary:
            sentence_text = str(sentence)
            sentence_word_count = len(sentence_text.split())

            if word_count + sentence_word_count <= target_word_count:
                summary_sentences.append(sentence_text)
                word_count += sentence_word_count
            elif not summary_sentences and sentence_word_count >= min_sentence_word_count:
                # If no sentences have been added and the sentence meets min_sentence_word_count, add it
                summary_sentences.append(sentence_text)
                break
            else:
                break  # Stop if we exceed the target word count

        return " ".join(summary_sentences)  # Return summary as a string

    # Replace any user-specified "\n" with actual line breaks
    text = txt.replace("\\n", "\n")

    # Summarize the text with the target word count
    return summarize_text(text, target_word_count, min_sentence_word_count)


def abstractive_summarization_bart(text, target_word_count, min_sentence_word_count=10):
    from transformers import pipeline
    import torch  # Import torch for checking GPU availability

    # Check if GPU is available (device 0 for GPU, -1 for CPU)
    device = 0 if torch.cuda.is_available() else -1

    # Initialize the BART summarization pipeline and specify the device
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=device)

    # Estimate token lengths to encourage complete sentences
    max_length = int(target_word_count * 2)  # Increase length to encourage full sentences
    min_length = max(int(target_word_count * 0.8), min_sentence_word_count)  # Ensure sentences meet min_sentence_word_count

    # Generate the summary
    summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)

    # Extract the summary text from the output
    summary_text = summary[0]['summary_text']

    # Ensure the output ends at a full stop
    if not summary_text.endswith('.'):
        last_period_index = summary_text.rfind('.')
        if last_period_index != -1:
            summary_text = summary_text[:last_period_index + 1]  # Trim to last complete sentence

    return summary_text  # Return summary as a string


def abstractive_summarization_llm(txt, target_word_count, min_sentence_word_count=10):
    import os
    os.environ["GOOGLE_API_KEY"] = os.getenv("MY_KEY")  # Replace with your actual API key

    # Ensure the necessary libraries are installed
    # Commented out for execution in environments where packages are already installed

    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.prompts import ChatPromptTemplate

    # Generalized function to load LLM (Gemini Models)
    def load_llm(model="gemini-1.5-flash"):
        llm = ChatGoogleGenerativeAI(
            model=model,
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2
        )
        return llm

    # Generalized function to get a prompt template
    def get_prompt_template():
        # Define prompt
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "Write a concise summary of the following in {num_words} words:\n\n"),
                ("human", "{context}")
            ]
        )
        return prompt

    # Function to summarize text using Google Gemini Models
    def summarize_text(text, target_word_count, min_sentence_word_count, model="gemini-1.5-flash"):
        llm = load_llm(model)
        prompt = get_prompt_template()
        chain = prompt | llm

        result = chain.invoke({
            "context": text,
            "num_words": target_word_count
        })

        # Get the generated summary
        summary = result.content

        # Ensure the summary ends at a full stop and within the target word count
        summary_words = summary.split()

        # If summary is longer than the target word count, trim it at the closest sentence boundary
        if len(summary_words) > target_word_count:
            summary = ' '.join(summary_words[:target_word_count])

            # Find the last full stop to ensure the summary ends at a complete sentence
            last_period_index = summary.rfind('.')
            if last_period_index != -1:
                summary = summary[:last_period_index + 1]  # Trim to the last complete sentence
            else:
                # If no full stop found, trim to the nearest sentence boundary
                sentence_endings = ['.', '!', '?']
                for char in sentence_endings:
                    last_index = summary.rfind(char)
                    if last_index != -1:
                        summary = summary[:last_index + 1]
                        break

        elif len(summary_words) < min_sentence_word_count:
            # If the summary is shorter than the minimum sentence word count, return the full summary
            return summary

        return summary  # Return summary as a string

    # Example text for summarization
    text = txt

    # Generate and return the summary with target word count
    summary = summarize_text(text, target_word_count, min_sentence_word_count, model="gemini-1.5-flash")

    return summary  # Return summary as a string


def abstractive_summarization_t5(txt, target_word_count, min_sentence_word_count=10):
    from transformers import T5Tokenizer, T5ForConditionalGeneration

    # Load the pre-trained T5 model and tokenizer from Hugging Face
    model_name = "t5-small"
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    tokenizer = T5Tokenizer.from_pretrained(model_name)

    def summarize_text(text, target_word_count, min_sentence_word_count):
        # Prepend "summarize:" to the input text
        input_text = "summarize: " + text

        # Tokenize the input text
        inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)

        # Generate the summary (using beam search for improved quality)
        summary_ids = model.generate(inputs, max_length=150, min_length=40,
                                     length_penalty=2.0, num_beams=4, early_stopping=True)

        # Decode the generated tokens into text
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        # Ensure the summary ends near the target word count
        summary_words = summary.split()

        # If the summary is longer than the target word count, truncate it properly at a full stop
        if len(summary_words) > target_word_count:
            summary = ' '.join(summary_words[:target_word_count])

            # Find the last full stop to ensure we don't cut off mid-sentence
            last_period_index = summary.rfind('.')
            if last_period_index != -1:
                summary = summary[:last_period_index + 1]  # Trim to the last complete sentence
            else:
                # If no full stop found, trim to the nearest sentence boundary (period, exclamation mark, or question mark)
                sentence_endings = ['.', '!', '?']
                for char in sentence_endings:
                    last_index = summary.rfind(char)
                    if last_index != -1:
                        summary = summary[:last_index + 1]
                        break

        elif len(summary_words) < min_sentence_word_count:
            # If the summary is shorter than the minimum sentence word count, return the full summary
            return summary

        return summary  # Return summary as a string

    # Generate and return the summary with target word count
    summary = summarize_text(txt, target_word_count, min_sentence_word_count)

    return summary  # Return summary as a string

def abstractive_summarization_langchain_iterate(pdf, n):
    import os
    os.environ['GOOGLE_API_KEY'] = os.getenv("MY_KEY")

    from langchain_community.document_loaders import PyPDFLoader
    from langchain.chains.summarize import load_summarize_chain

    from langchain_google_genai import ChatGoogleGenerativeAI

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2
    )

    def summarize_pdf(pdf_file_path):
        loader = PyPDFLoader(pdf_file_path)
        docs = loader.load_and_split()
        chain = load_summarize_chain(llm, chain_type="refine")
        summary = chain.invoke(docs)

        return summary

    summary = summarize_pdf(pdf)

    return summary['output_text']

def abstractive_summarization_langchain_map(pdf, n):
    import os
    os.environ['GOOGLE_API_KEY'] = os.getenv("MY_KEY")

    from langchain_community.document_loaders import PyPDFLoader
    from langchain.chains.summarize import load_summarize_chain

    from langchain_google_genai import ChatGoogleGenerativeAI

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2
    )

    def summarize_pdf(pdf_file_path):
        loader = PyPDFLoader(pdf_file_path)
        docs = loader.load_and_split()
        chain = load_summarize_chain(llm, chain_type="map_reduce")
        summary = chain.invoke(docs)

        return summary

    summary = summarize_pdf(pdf)

    return summary['output_text']


def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF file."""
    reader = PdfReader(pdf_file.name)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Define placeholder functions for summarization
def extractive_summarize_text(text, method, n):
    if method == "LSA":
        return extractive_summarization_lsa(text,n)
    elif method == "TFIDF":
        return extractive_summarization_tfidf(text,n)
    elif method == "FREQUENCY":
        return extractive_summarization_frequency(text,n)
    else:
        return "Please select a valid summarization method."

def abstractive_summarize_text(text, method, n):
    if method == "BART":
        return abstractive_summarization_bart(text,n)
    elif method == "LLM":
        return abstractive_summarization_llm(text,n)
    elif method == "T5":
        return abstractive_summarization_t5(text,n)
    else:
        return "Please select a valid summarization method."

def extractive_handler(pdf_file, input_text, method, word_count):
    text = input_text
    if pdf_file is not None:
        text = extract_text_from_pdf(pdf_file)
    return extractive_summarize_text(text, method, word_count)

def abstractive_handler(pdf_file, input_text, method, word_count):
    text = input_text
    pdf = pdf_file
    n = word_count
    if pdf_file is not None:
        if method == "Langchain-map":
            return abstractive_summarization_langchain_map(pdf, n)
        elif method == "Langchain-iterate":
            return abstractive_summarization_langchain_iterate(pdf, n)
        else:
            text = extract_text_from_pdf(pdf_file)
    return abstractive_summarize_text(text, method, word_count)


import gradio as gr
from PyPDF2 import PdfReader  # Install this package: pip install PyPDF2

css = """
h1 {
    margin-top: 2rem;
    font-size: 2rem;
    text-align: center;
}
generate-btn {
    background-color: orange !important;
    color: white !important;
    border: none !important;
    font-weight: bold;
}
"""

with gr.Blocks(title="Summarizer App", css=css) as demo:
    gr.Markdown("# Summarizer App")

    # Main layout: input controls on the left, output on the right
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Input Selection")

            # Input type selection
            input_type = gr.Radio(
                choices=["Text Input", "Upload PDF"],
                label="Select Input Type",
                value="Text Input"
            )
            
            # Summarization method selection
            method_type = gr.Radio(
                choices=["Extractive", "Abstractive"],
                label="Select Summarization Method",
                value="Extractive"
            )

            # Input fields
            input_text = gr.Text(label="Input Text", lines=10, visible=True)
            pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"], visible=False)
            
            # Word count and algorithm selection
            summarization_method = gr.Dropdown(
                choices=["LSA", "FREQUENCY", "TFIDF", "BART", "LLM", "T5", "Langchain-map", "Langchain-iterate"],
                label="Select Algorithm"
            )
            word_count_input = gr.Number(value=50, label="Number of Words for Summary", precision=0)

            # Update input field visibility based on input type
            def update_input_type(input_type):
                if input_type == "Text Input":
                    return gr.update(visible=True), gr.update(visible=False)
                else:
                    return gr.update(visible=False), gr.update(visible=True)

            input_type.change(
                update_input_type,
                inputs=[input_type],
                outputs=[input_text, pdf_input]
            )

            # Update summarization algorithm choices based on method type
            def update_method_type(method_type):
                if method_type == "Extractive":
                    return gr.update(choices=["LSA", "FREQUENCY", "TFIDF"], value="LSA")
                else:
                    return gr.update(choices=["BART", "LLM", "T5", "Langchain-map", "Langchain-iterate"], value="BART")

            method_type.change(
                update_method_type,
                inputs=[method_type],
                outputs=[summarization_method]
            )

            # Buttons for Generate and Clear
            with gr.Row():
                clear_button = gr.Button("Clear All")
                generate_button = gr.Button("Generate Summary")
        
        # Output area
        with gr.Column():
            gr.Markdown("### Summary Output")
            output_text = gr.Textbox(label="Summary Output", lines=10, interactive=False)

    # Generate summary handler
    def generate_summary(method_type, pdf_file, input_text, method, word_count):
        if method_type == "Extractive":
            return extractive_handler(pdf_file, input_text, method, word_count)
        else:
            return abstractive_handler(pdf_file, input_text, method, word_count)

    # Clear fields handler
    def clear_fields():
        return None, None, None

    # Link buttons to handlers
    generate_button.click(
        generate_summary,
        inputs=[method_type, pdf_input, input_text, summarization_method, word_count_input],
        outputs=[output_text]
    )

    clear_button.click(
        clear_fields,
        inputs=[],
        outputs=[pdf_input, input_text, output_text]
    )

demo.launch()


"""input_text = gr.Text(label="Input Text", lines=10)
word_count_input = gr.Number(value=50, label="Number of Words for Summary", precision=0)
pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"])


with gr.Blocks(title="Summarizer App", css=css) as demo:
    gr.Markdown("# Summarizer App")

    with gr.Tabs():
        with gr.TabItem("Extractive"):
            gr.Interface(
                fn=extractive_handler,
                inputs=[
                    pdf_input, input_text,
                    gr.Dropdown(choices=["LSA", "FREQUENCY", "TFIDF"], label="Select Method"),
                    word_count_input  # Added input for word count
                ],
                outputs=['text'],
                flagging_mode='never',
                submit_btn='Generate'
            )
        with gr.TabItem("Abstractive"):
            gr.Interface(
                fn=abstractive_handler,
                inputs=[
                    pdf_input, input_text,
                    gr.Dropdown(choices=[ "BART", "LLM", "T5", "Langchain"], label="Select Method"),
                    word_count_input  # Added input for word count
                ],
                outputs=['text'],
                flagging_mode='never',
                submit_btn='Generate'
            )

demo.launch()"""




"""with gr.Blocks(title="Summarizer App", css=css) as demo:
    gr.Markdown("# Summarizer App")

    with gr.Tabs():
        with gr.TabItem("Extractive"):
            with gr.Row():
                input_text = gr.Text(label="Input Text", lines=10)
                pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"])
            
            word_count_input = gr.Number(value=50, label="Number of Words for Summary", precision=0)
            dropdown_method = gr.Dropdown(choices=["LSA", "FREQUENCY", "TFIDF"], label="Select Method")

            def extractive_handler(pdf_file, input_text, method, word_count):
                text = input_text
                if pdf_file is not None:
                    text = extract_text_from_pdf(pdf_file)
                return extractive_summarize_text(text, method, word_count)

            gr.Interface(
                fn=extractive_handler,
                inputs=[pdf_input, input_text, dropdown_method, word_count_input],
                outputs=['text'],
                flagging_mode='never',
                submit_btn='Generate'
            )
        
        with gr.TabItem("Abstractive"):
            with gr.Row():
                input_text = gr.Text(label="Input Text", lines=10)
                pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"])
            
            word_count_input = gr.Number(value=50, label="Number of Words for Summary", precision=0)
            dropdown_method = gr.Dropdown(choices=["BART", "LLM", "T5", "Langchain"], label="Select Method")

            def abstractive_handler(pdf_file, input_text, method, word_count):
                text = input_text
                pdf = pdf_file
                n = word_count
                if pdf_file is not None:
                    if method == "Langchain":
                        return abstractive_summarization_langchain(pdf, n)
                    else:
                        text = extract_text_from_pdf(pdf_file)
                return abstractive_summarize_text(text, method, word_count)

            gr.Interface(
                fn=abstractive_handler,
                inputs=[pdf_input, input_text, dropdown_method, word_count_input],
                outputs=['text'],
                flagging_mode='never',
                submit_btn='Generate'
            )

demo.launch()"""

