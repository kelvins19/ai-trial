from typing import List, Dict
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
import nest_asyncio
import asyncio
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import glob
import faiss
from sentence_transformers import SentenceTransformer

load_dotenv()

nest_asyncio.apply()

class Chatbot:
    def __init__(self, model_name: str = None, api_key: str = None, base_url: str = None, prompt: str = None, pdf_dir: str = None):
        # Use CONFIG constant for default values
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
            
        self.model_name = model_name or os.getenv("OPENAI_MODEL_NAME")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        
        self.model = OpenAIModel(
            model_name=self.model_name, 
            api_key=self.api_key,
            base_url=self.base_url,
        )

        system_prompt = "You are an AI assistant with expert knowledge derived from the provided PDF document."
        if prompt != "": 
            system_prompt = prompt

        system_prompt += " Answer questions based on the information from the PDF document, providing detailed and accurate responses."

        self.agent = Agent(self.model, system_prompt=system_prompt)
        self.pdf_dir = pdf_dir
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.contents = self._load_pdf_contents() if pdf_dir else []
        self.vector_db = self._create_vector_db() if pdf_dir else None

    def _load_pdf_contents(self) -> List[str]:
        contents = []
        pdf_paths = glob.glob(os.path.join(self.pdf_dir, "*.pdf"))
        for pdf_path in pdf_paths:
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")

            with open(pdf_path, "rb") as file:
                reader = PdfReader(file)
                for page_num in range(len(reader.pages)):
                    page = reader.pages[page_num]
                    contents.append(page.extract_text())
        return contents

    def _create_vector_db(self):
        vectors = self.sentence_model.encode(self.contents)
        dimension = vectors.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(vectors)
        return index

    def _search_vector_db(self, query: str) -> str:
        query_vector = self.sentence_model.encode([query])
        D, I = self.vector_db.search(query_vector, k=1)
        return self.contents[I[0][0]]

    async def _call_openrouter_batch(self, prompt: str) -> List[str]:
        try:
            # Print debug information
            print(f"Calling OpenRouter API with model: {self.model_name}")
            print(f"Using base URL: {self.base_url}")

            relevant_text = self._search_vector_db(prompt)
            full_prompt = relevant_text + "\n\n" + prompt
            result = await self.agent.run(user_prompt=full_prompt)

            total_tokens = result.usage().total_tokens

            print(f"Total tokens used: {total_tokens}")

            return result.data

        except Exception as e:
            print(f"API Error detail: {str(e)}")  # Debug print
            raise Exception(f"API Error: {str(e)}")
