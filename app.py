# Import necessary libraries
import PyPDF2
import requests  # Make sure to install the requests library
import time  # Import time for sleep functionality
import re
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, GPT2LMHeadModel, BlenderbotTokenizer, BlenderbotForConditionalGeneration, GPT2Tokenizer
import nltk
import ssl
import torch
import threading
import queue
from functools import lru_cache
import asyncio
from concurrent.futures import ThreadPoolExecutor

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("NLTK data is missing. Downloading required data...")
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    print("Download complete.")

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Your Hugging Face API key
API_KEY = "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"  # Replace with your actual API key

class PDFQuestionAnswering:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.pdf_text = self.extract_and_clean_text()
        self.qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")
        self.stop_words = set(stopwords.words('english'))
        
        # Initialize GPT-2 model for AI explanations
        model_name = "gpt2-medium"
        self.ai_tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.ai_model = GPT2LMHeadModel.from_pretrained(model_name)
        self.ai_tokenizer.pad_token = self.ai_tokenizer.eos_token
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.ai_model.to(self.device)

    def extract_and_clean_text(self):
        try:
            with open(self.pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = " ".join(page.extract_text() for page in reader.pages)
            return ' '.join(text.split())  # Clean the text
        except Exception as e:
            print(f"Error extracting text from PDF: {e}")
            return ""

    def preprocess_question(self, question):
        word_tokens = word_tokenize(question.lower())
        return [w for w in word_tokens if w not in self.stop_words]

    def find_relevant_context(self, question, window_size=2000):
        processed_question = self.preprocess_question(question)
        words = self.pdf_text.split()
        best_context = ""
        max_overlap = 0

        for i in range(0, len(words), window_size // 2):
            context = ' '.join(words[i:i+window_size])
            context_words = set(context.lower().split())
            overlap = len(set(processed_question).intersection(context_words))
            if overlap > max_overlap:
                max_overlap = overlap
                best_context = context

        return best_context if best_context else self.pdf_text[:window_size]

    @lru_cache(maxsize=150)
    def generate_ai_explanation(self, question):
        print("Generating explanation... This may take up to 30 seconds.")
        start_time = time.time()

        try:
            prompt = f"Question: What is a {question}?\nAnswer: A {question} is "
            input_ids = self.ai_tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=self.device)
            
            output = self.ai_model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=200,
                num_return_sequences=1,
                no_repeat_ngram_size=2,
                temperature=0.7,
                top_k=50,
                top_p=0.95,
                do_sample=True,
                pad_token_id=self.ai_tokenizer.eos_token_id
            )
            
            generated_text = self.ai_tokenizer.decode(output[0], skip_special_tokens=True)
            answer = generated_text.split("Answer: ")[-1].strip()
            
            # Post-process the answer
            sentences = answer.split('.')
            relevant_sentences = [s for s in sentences if question.lower() in s.lower()]
            if relevant_sentences:
                answer = '. '.join(relevant_sentences) + '.'
            else:
                answer = sentences[0] + '.'  # Take at least the first sentence if no relevant ones found
            
            end_time = time.time()
            print(f"Explanation generated in {end_time - start_time:.2f} seconds.")
            
            return answer
        except Exception as e:
            print(f"Error generating explanation: {e}")
            return "I apologize, but I'm unable to provide an explanation at the moment. Please try asking another question."

    def generate_additional_info(self, question):
        prompt = f"Provide more detailed information about {question}:\n\n"
        input_ids = self.ai_tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        output = self.ai_model.generate(
            input_ids,
            max_length=150,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            do_sample=True,
            pad_token_id=self.ai_tokenizer.eos_token_id
        )
        additional_info = self.ai_tokenizer.decode(output[0], skip_special_tokens=True)[len(prompt):].strip()
        return additional_info

    def answer_question(self, question):
        if question.lower() in ["hi", "hello", "what's up", "how are you"]:
            return "Hello! I'm here to answer questions about the PDF content or provide general information. How can I help you?"

        # Check if the question is about the PDF content
        question_words = set(word.lower() for word in question.split())
        pdf_words = set(word.lower() for word in self.pdf_text.split())
        is_pdf_related = bool(question_words.intersection(pdf_words))

        if not is_pdf_related:
            print("Content not found in PDF. Providing a general explanation...")
            return self.generate_ai_explanation(f"Provide a detailed explanation about: {question}")

        relevant_context = self.find_relevant_context(question)
        print(f"\nContext used: {relevant_context[:200]}...\n")

        result = self.qa_pipeline(question=question, context=relevant_context)

        if result['score'] > 0.7:
            return f"Answer from PDF: {result['answer']} (Confidence: {result['score']:.2f})"
        else:
            print("No clear answer found in PDF. Providing a general explanation...")
            return self.generate_ai_explanation(f"Provide a detailed explanation about: {question}")

def main():
    pdf_path = r'C:\Users\deeps\Downloads\CN MSE.pdf'  # Update this path
    qa_system = PDFQuestionAnswering(pdf_path)

    print("\nPDF Question Answering System")
    print("Type 'exit' to quit the program.")

    while True:
        question = input("Ask a question: ")
        if question.lower() == 'exit':
            break

        print("Processing your question... Please wait.")
        answer = qa_system.answer_question(question)
        print(f"\nAnswer: {answer}\n")

if __name__ == "__main__":
    main()
