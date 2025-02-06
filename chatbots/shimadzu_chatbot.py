import sys
import os
import subprocess
import platform

print("Python executable:", sys.executable)

from sentence_transformers import SentenceTransformer, util
from PyPDF2 import PdfReader

class PDFKnowledgeBase:
    def __init__(self, pdf_dir):
        if not os.path.isdir(pdf_dir):
            raise NotADirectoryError(f"{pdf_dir} is not a directory")
        self.pdf_dir = pdf_dir
        self.knowledge = self._extract_knowledge()
        if not self.knowledge:
            raise ValueError("No PDF files found or extracted text is empty")

    def _extract_knowledge(self):
        knowledge = ""
        for filename in os.listdir(self.pdf_dir):
            if filename.endswith('.pdf'):
                filepath = os.path.join(self.pdf_dir, filename)
                knowledge += self._extract_text_from_pdf(filepath)
        return knowledge

    def _extract_text_from_pdf(self, filepath):
        text = ""
        with open(filepath, 'rb') as file:
            reader = PdfReader(file)
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text()
        return text

class ChatBot:
    def __init__(self, knowledge_base):
        self.knowledge_base = knowledge_base
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.knowledge_embeddings = self.model.encode([self.knowledge_base.knowledge], convert_to_tensor=True)

    def answer_question(self, question):
        question_embedding = self.model.encode(question, convert_to_tensor=True)
        scores = util.pytorch_cos_sim(question_embedding, self.knowledge_embeddings)[0]
        best_score_idx = scores.argmax().item()
        response = self.knowledge_base.knowledge.split('\n')[best_score_idx]
        return response

# Usage
pdf_dir = './assets'  # Update this to the correct directory containing PDF files
knowledge_base = PDFKnowledgeBase(pdf_dir)
chatbot = ChatBot(knowledge_base)

question = "What is the main topic of the first document?"
question = "Explain me about UV detection method"
answer = chatbot.answer_question(question)
print("================================================")
print(answer)
