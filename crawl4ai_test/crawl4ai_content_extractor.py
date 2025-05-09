import json
import requests
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
import nest_asyncio
from utils.timer import Timer
import asyncio
import time as time_module 
from pydantic import BaseModel, Field, field_validator
import os
from dotenv import load_dotenv
from collections import defaultdict

load_dotenv()
nest_asyncio.apply()

class Deps(BaseModel):
    prompt: str

class ContentExtractor:
    def __init__(self, model_name: str = None, api_key: str = None, base_url: str = None):
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

        # System prompt for service extraction
        self.service_extraction_prompt = """You are a service extraction assistant. Your task is to:
        1. Extract services from the input text
        2. For each service, identify:
           - Service name
           - Duration in minutes (default to 60 if not specified)
           - Service description
        3. Output the results in JSON format like this:
        [{"name": "SERVICE_NAME", "duration_in_minutes": 60, "description": "SERVICE_DESC"}]
        4.  Ensure the output is valid JSON"""

        # System prompt for content extraction
        self.content_extraction_prompt = """You are a content extraction assistant. Your task is to:
        1. Extract and clean up the main content from the input text, which may contain multiple types of content (FAQs, location info, contact info, etc.)
        2. For each piece of content:
           - Generate a concise title that summarizes the content
           - Categorize the content into one of these categories:
             * FAQ: For question-answer pairs or frequently asked questions
             * General: For location info, contact info, or general business information
             * Other: For any content that doesn't fit the above categories
           - Determine the language (EN for English, BI for Bahasa Indonesia, CN for Mandarin)
        3. Output the results in JSON format like this:
        [{"category": "FAQ", "title": "AI_GENERATED_TITLE", "content": "CONTENT", "language": "EN"}]
        4. Ensure the output is valid JSON"""

        self.timer = Timer()
        self.sessions = defaultdict(dict)
        self.agents = {}

    def get_agent(self, is_service_extraction: bool = True) -> Agent:
        agent_key = f"{'service' if is_service_extraction else 'content'}"
        if agent_key not in self.agents:
            print("--------------------------------")
            print(f"Agent Session not found: {agent_key}")
            print("--------------------------------")
            system_prompt = self.service_extraction_prompt if is_service_extraction else self.content_extraction_prompt
            agent = Agent(self.model, system_prompt=system_prompt, deps_type=Deps)
            self.agents[agent_key] = agent
        return self.agents[agent_key]

    async def _call_openrouter_batch(self, deps: Deps, is_service_extraction: bool = True) -> str:
        max_retries = 3
        retry_delay = 1

        for attempt in range(max_retries):
            try:
                print(f"Calling OpenRouter API with model: {self.model_name}")
                print(f"Using base URL: {self.base_url}")

                print(f"User prompt: {deps.prompt}")
                self.timer.start()
                agent = self.get_agent(is_service_extraction)

                result = await agent.run(user_prompt=deps.prompt, deps=deps)
                self.timer.stop()

                if result is None:
                    raise Exception("API returned no result")

                usage = result.usage()
                if usage is None:
                    raise Exception("API usage information is missing")

                total_tokens = usage.total_tokens
                total_time = self.timer.get_elapsed_time()

                print(f"Total tokens used: {total_tokens}")
                print(f"Total time taken: {total_time:.3f} ms")

                response = result.data

                print(f"API Response: {response}")

                # Clean up the response
                if response.endswith("```}"):
                    response = response.replace("```}", "").strip()
                if response.endswith("```"):
                    response = response.replace("```", "").strip()
                if response.startswith("```"):
                    response = response.replace("```", "").strip()
                if response.startswith("json"):
                    response = response.replace("json", "").strip()

                return response

            except Exception as e:
                print(f"API Error detail: {str(e)}")
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...")
                    time_module.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    raise Exception(f"API Error: {str(e)}")

    async def extract_services(self, input_prompt: str) -> str:
        """Extract services from the input prompt"""
        prompt = f"""
        {input_prompt}
        """

        crawlDeps = Deps(
            prompt=prompt
        )

        response = await self._call_openrouter_batch(crawlDeps, is_service_extraction=True)
        return response

    async def extract_content(self, input_prompt: str) -> str:
        """Extract and clean up content from the input prompt"""
        prompt = f"""
        {input_prompt}
        """

        crawlDeps = Deps(
            prompt=prompt
        )

        response = await self._call_openrouter_batch(crawlDeps, is_service_extraction=False)
        return response

async def main():
    # Initialize the ContentExtractor
    extractor = ContentExtractor(
        model_name=os.getenv("OPENAI_MODEL_NAME"),
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL")
    )

    # Example prompts
    service_prompt = """
    Our spa offers the following services:
    1. Swedish Massage - 60 minutes of relaxation
    2. Deep Tissue Massage - 90 minutes of therapeutic treatment
    3. Facial Treatment - Rejuvenating your skin
    4. Hot Stone Massage - 75 minutes of deep relaxation
    """

    content_prompt = """
    Welcome to our spa! We are located in the heart of the city and offer various treatments.
    Our experienced therapists will ensure you have a relaxing experience.
    Please book your appointment in advance.
    """

    # Test service extraction
    print("\nTesting Service Extraction:")
    service_response = await extractor.extract_services(service_prompt)
    print(f"Service extraction response: {service_response}")

    # Test content extraction
    print("\nTesting Content Extraction:")
    content_response = await extractor.extract_content(content_prompt)
    print(f"Content extraction response: {content_response}")

if __name__ == "__main__":
    asyncio.run(main()) 