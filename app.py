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
import signal
import json
import os
import sys
from collections import Counter

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
API_KEY = "Replace with your actual API key"  # Replace with your actual API key

class TimeoutException(Exception): pass

def timeout_handler(signum, frame):
    raise TimeoutException("Explanation generation timed out")

class PDFQuestionAnswering:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.pdf_text = self.extract_and_clean_text()
        self.qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")
        self.stop_words = set(stopwords.words('english'))
        
        # Initialize GPT-2 model for AI explanations
        model_name = "gpt2-large"  # Using a larger model for better quality
        self.ai_tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.ai_model = GPT2LMHeadModel.from_pretrained(model_name)
        self.ai_tokenizer.pad_token = self.ai_tokenizer.eos_token
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.ai_model.to(self.device)

        # Initialize explanation cache
        self.cache_file = "explanation_cache.json"
        self.explanation_cache = self.load_cache()

        self.common_definitions = {
            "phone": "A phone is a communication device used for voice calls and, in modern smartphones, text messaging, internet access, and running applications.",
            "computer": "A computer is an electronic device that processes data according to a set of instructions (programs), capable of storing, retrieving, and processing data.",
            "internet": "The Internet is a global network of interconnected computers that allows for the exchange of information and communication across the world.",
            "email": "Email (electronic mail) is a method of exchanging digital messages between people using digital devices such as computers, tablets, and smartphones.",
            # Add more common definitions here
        }

    def extract_and_clean_text(self):
        try:
            with open(self.pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = " ".join(page.extract_text() for page in reader.pages if page.extract_text())
            return ' '.join(text.split())  # Clean the text
        except Exception as e:
            print(f"Error extracting text from PDF: {e}")
            return ""

    def preprocess_question(self, question):
        # Convert to lowercase and remove punctuation
        question = re.sub(r'[^\w\s]', '', question.lower())
        # Tokenize and remove stop words
        return [word for word in question.split() if word not in self.stop_words]

    def find_relevant_context(self, question, context_size=2000):
        # Preprocess the question
        question_words = set(self.preprocess_question(question))
        
        # Create a list of related terms for networking concepts
        related_terms = {
            'ipv4': ['ip', 'internet protocol', 'header', 'packet', 'addressing'],
            'csma': ['carrier sense', 'multiple access', 'collision', 'ethernet'],
            # Add more related terms for other networking concepts
        }
        
        # Extend question words with related terms
        for term, related in related_terms.items():
            if term in question_words:
                question_words.update(related)
        
        # Split the PDF text into sentences
        sentences = re.split(r'(?<=[.!?])\s+', self.pdf_text)
        
        # Score each sentence based on the number of question words it contains
        sentence_scores = []
        for i, sentence in enumerate(sentences):
            words = set(word.lower() for word in re.findall(r'\w+', sentence))
            score = len(question_words.intersection(words))
            sentence_scores.append((i, score))
        
        # Sort sentences by score in descending order
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Select top scoring sentences
        top_sentences = sentence_scores[:10]  # Adjust this number as needed
        
        # Sort selected sentences by their original order in the document
        top_sentences.sort(key=lambda x: x[0])
        
        # Construct the context from the selected sentences and their neighbors
        context = []
        for idx, _ in top_sentences:
            start = max(0, idx - 2)
            end = min(len(sentences), idx + 3)
            context.extend(sentences[start:end])
        
        # Join the context sentences
        return ' '.join(context)

    def load_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'r') as f:
                return json.load(f)
        return {}

    def save_cache(self):
        with open(self.cache_file, 'w') as f:
            json.dump(self.explanation_cache, f)

    def generate_ai_explanation(self, question):
        yield "Generating a focused explanation...\n"
        start_time = time.time()

        # Check if we have a pre-defined answer
        lower_question = question.lower()
        for key, value in self.common_definitions.items():
            if key in lower_question:
                yield f"Predefined explanation found in {time.time() - start_time:.2f} seconds.\n\n"
                yield value
                return

        try:
            prompt = f"Explain the concept of '{question}' clearly and concisely in 2-3 sentences, focusing on its primary meaning or function:\n\n"
            
            input_ids = self.ai_tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=self.device)
            
            output = self.ai_model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=150,
                num_return_sequences=1,
                no_repeat_ngram_size=2,
                temperature=0.7,
                top_k=50,
                top_p=0.95,
                do_sample=True,
                pad_token_id=self.ai_tokenizer.eos_token_id
            )
            
            generated_text = self.ai_tokenizer.decode(output[0], skip_special_tokens=True)
            explanation = generated_text[len(prompt):].strip()
            
            # Post-process the explanation
            explanation = re.sub(r'\s+', ' ', explanation)  # Remove extra whitespace
            sentences = explanation.split('.')
            relevant_sentences = [s.strip() for s in sentences if question.lower() in s.lower() or len(s.split()) > 5]
            processed_explanation = '. '.join(relevant_sentences[:3])  # Limit to 3 sentences max
            
            if not processed_explanation or len(processed_explanation.split()) < 10:
                processed_explanation = f"\n\n{question.capitalize()} is a term that may require more context. Could you please provide more details or rephrase your question?"
            
            if not processed_explanation.endswith('.'):
                processed_explanation += '.'
            
            yield f"\n\n{processed_explanation}"
            yield f"\n\nExplanation generated in {time.time() - start_time:.2f} seconds."

        except Exception as e:
            yield f"\n\nError generating explanation: {e}"

    def _generate_text(self, prompt, max_length=200):
        input_ids = self.ai_tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=self.device)
        
        output = self.ai_model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            num_return_sequences=1,
            no_repeat_ngram_size=3,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            do_sample=True,
            pad_token_id=self.ai_tokenizer.eos_token_id
        )
        
        generated_text = self.ai_tokenizer.decode(output[0], skip_special_tokens=True)
        return generated_text[len(prompt):].strip()

    def _refine_text(self, text):
        refine_prompt = f"Refine and improve the following explanation, ensuring it's coherent, accurate, and well-structured:\n\n{text}\n\nRefined explanation:"
        refined_text = self._generate_text(refine_prompt, max_length=500)
        return refined_text

    def generate_additional_info(self, question):
        prompt = f"Provide additional important information about {question} that hasn't been mentioned yet:\n\n"
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
        
        additional_info = self.ai_tokenizer.decode(output[0], skip_special_tokens=True)[len(prompt):].strip()
        return additional_info

    def answer_question(self, question):
        relevant_context = self.find_relevant_context(question)
        if relevant_context:
            print(f"\nContext used (first 200 characters): {relevant_context[:200]}...\n")
            print(f"Total context length: {len(relevant_context)} characters\n")
            
            # Extract key information from the context
            key_info = self.extract_key_information(question, relevant_context)
            
            if key_info:
                return f"Answer from PDF:\n{key_info}"
            else:
                # If no key information extracted, use the QA pipeline
                result = self.qa_pipeline(question=question, context=relevant_context)
                if result['score'] > 0.5:
                    return f"Answer from PDF: {result['answer']} (Confidence: {result['score']:.2f})"
                else:
                    print(f"Low confidence answer found (score: {result['score']:.2f}). Generating AI explanation...")
        else:
            print("No relevant context found in the PDF. Generating AI explanation...")
        
        return None

    def extract_key_information(self, question, context):
        # Define patterns for different types of information
        patterns = {
            'ipv4': r'IPv4 header.*?(?=\n\n|\Z)',
            'csma': r'CSMA.*?(?=\n\n|\Z)',
            # Add more patterns for other topics
        }
        
        # Find the relevant pattern based on the question
        relevant_pattern = None
        for key, pattern in patterns.items():
            if key in question.lower():
                relevant_pattern = pattern
                break
        
        if relevant_pattern:
            match = re.search(relevant_pattern, context, re.DOTALL | re.IGNORECASE)
            if match:
                info = match.group(0)
                # Clean up the extracted information
                info = re.sub(r'\s+', ' ', info).strip()
                # Split into bullet points if possible
                info_points = info.split('●')
                if len(info_points) > 1:
                    return '\n'.join(['●' + point.strip() for point in info_points if point.strip()])
                else:
                    return info
        
        return None

    def generate_detailed_explanation(self, topic):
        print(f"Generating a detailed explanation for '{topic}'... This may take up to 1 minute.")
        start_time = time.time()

        try:
            prompt = f"Provide a detailed explanation of '{topic}' in 4-5 sentences. Include its definition, key features, common uses, and any other relevant information:\n\n"
            
            input_ids = self.ai_tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=self.device)
            
            output = self.ai_model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=300,
                num_return_sequences=1,
                no_repeat_ngram_size=2,
                temperature=0.7,
                top_k=50,
                top_p=0.95,
                do_sample=True,
                pad_token_id=self.ai_tokenizer.eos_token_id
            )
            
            generated_text = self.ai_tokenizer.decode(output[0], skip_special_tokens=True)
            explanation = generated_text[len(prompt):].strip()
            
            # Post-process the explanation
            explanation = re.sub(r'\s+', ' ', explanation)  # Remove extra whitespace
            sentences = explanation.split('.')
            relevant_sentences = [s.strip() for s in sentences if topic.lower() in s.lower() or len(s.split()) > 5]
            processed_explanation = '. '.join(relevant_sentences[:5])  # Limit to 5 sentences max
            
            if not processed_explanation or len(processed_explanation.split()) < 20:
                processed_explanation = f"I apologize, but I couldn't generate a detailed explanation for '{topic}'. This might be due to limited information or the complexity of the topic. Could you please provide more context or ask about a more specific aspect of {topic}?"
            
            if not processed_explanation.endswith('.'):
                processed_explanation += '.'
            
            end_time = time.time()
            print(f"Detailed explanation generated in {end_time - start_time:.2f} seconds.")

            return processed_explanation

        except Exception as e:
            print(f"Error generating detailed explanation: {e}")
            return f"I apologize, but I'm unable to provide a detailed explanation for '{topic}' at the moment. Please try asking another question or rephrasing your query."

def main():
    pdf_path = r'C:\Users\deeps\Downloads\CN MSE.pdf'  # Update this path
    qa_system = PDFQuestionAnswering(pdf_path)

    print("\nPDF Question Answering System")
    print("Type 'exit' to quit the program.")

    while True:
        question = input("\nAsk a question: ")
        if question.lower() == 'exit':
            break

        print("Processing your question... Please wait.")
        
        # First, try to answer from the PDF
        pdf_answer = qa_system.answer_question(question)
        if pdf_answer:
            print(pdf_answer)
        else:
            # If no answer from PDF, use AI explanation
            print("No specific answer found in the PDF. Generating an explanation...")
            for chunk in qa_system.generate_ai_explanation(question):
                sys.stdout.write(chunk)
                sys.stdout.flush()
                time.sleep(0.02)  # Slightly faster typing effect
        
        print("\n")

if __name__ == "__main__":
    main()
