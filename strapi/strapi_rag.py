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
from pydantic_ai.messages import ModelMessagesTypeAdapter

load_dotenv()
nest_asyncio.apply()

class Deps(BaseModel):
    phone_number: str
    prompt: str

class Crawl4AIRag:
    def __init__(self, model_name: str = None, api_key: str = None, base_url: str = None, prompt: str = None):
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

        system_prompt = "You are a support chatbot for a knowledge retrieval system."
        if prompt != "": 
            system_prompt = prompt

        system_prompt += """
        Users will send queries, and you need to retrieve relevant context from the knowledge base and summarize it in a user-friendly format.
        """

        self.system_prompt = system_prompt
        self.timer = Timer()
        self.sessions = defaultdict(dict)
        self.agents = {}

    def get_agent(self, phone_number: str) -> Agent:
        if phone_number not in self.agents:
            print("--------------------------------")
            print("Agent Session not found: %s" % phone_number)
            print("--------------------------------")
            agent = Agent(self.model, system_prompt=self.system_prompt, deps_type=Deps)
            agent.tool(self.get_context)
            self.agents[phone_number] = agent
        return self.agents[phone_number]

    def get_context(self, ctx: RunContext[str]) -> str:
        """Get context based on user query"""
        phone_number = ctx.deps.phone_number
        query = ctx.deps.prompt
        print(f"Phone number: {phone_number}")
        print(f"Query: {query}")
        username = os.getenv("BASIC_AUTH_USERNAME")
        password = os.getenv("BASIC_AUTH_PASSWORD")

        try:
            response = requests.post(
                "http://localhost:8000/knowledge/context-strapi-temp",
                json={"query": query, "max_document": 20},
                auth=(username, password)
            )
            response.raise_for_status()
            context_data = response.json()
            return f"Context retrieved: {json.dumps(context_data, indent=2)}"
        except Exception as e:
            print(f"Exception: {str(e)}")
            return f"Error retrieving context: {str(e)}"

    async def _call_openrouter_batch(self, deps: Deps) -> str:
        max_retries = 3
        retry_delay = 1

        for attempt in range(max_retries):
            try:
                print(f"Calling OpenRouter API with model: {self.model_name}")
                print(f"Using base URL: {self.base_url}")

                print(f"User prompt: {deps.prompt}")
                self.timer.start()
                agent = self.get_agent(deps.phone_number)
                
                message_history_json = self.sessions[deps.phone_number].get('chat_history', [])
                message_history = ModelMessagesTypeAdapter.validate_json(json.dumps(message_history_json))

                result = await agent.run(user_prompt=deps.prompt, deps=deps, message_history=message_history)
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

                if response.endswith("```}"):
                    response = response.replace("```}", "").strip()
                if response.endswith("```"):
                    response = response.replace("```", "").strip()
                if response.startswith("```"):
                    response = response.replace("```", "").strip()
                if response.startswith("json"):
                    response = response.replace("json", "").strip()

                self.sessions[deps.phone_number]['chat_history'] = json.loads(result.all_messages_json().decode('utf-8'))

                return response

            except Exception as e:
                print(f"API Error detail: {str(e)}")
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...")
                    time_module.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    raise Exception(f"API Error: {str(e)}")

    async def get_response(self, input_prompt: str, phone_number: str) -> str:
        if input_prompt == "":
            input_prompt = "get_context"
        
        prompt = f"""
        Session: {phone_number}

        {input_prompt}
        """

        crawlDeps = Deps(
            phone_number=phone_number,
            prompt=prompt
        )

        response = await self._call_openrouter_batch(crawlDeps)
        return response


async def main():
    # Initialize the Chatbot with the directory containing PDF files and other parameters
    chatbot = Crawl4AIRag(
        model_name=os.getenv("OPENAI_MODEL_NAME"),
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        prompt=""
    )

    prompt = "What is the customer service phone number of the mall?"
    # prompt = "What is the mall address?"
    prompt = "Is there any ongoing event?"
    # prompt = "Tell me about weekday dine and delight event"
    # prompt = "What is available deals on i12katong mall?"
    # prompt = "Where is watsons located in this mall?"
    # prompt = "What is rewards+?"
    response = await chatbot.get_response(input_prompt=prompt, phone_number="1234567890")
    
    # Print the response
    print("================================================")
    print(f"Chatbot response: {response}")
    
    # Print the chat history
    # chat_history = chatbot.get_chat_history(phone_number="1234567890")
    # print("Chat history:", chat_history)

# Run the test
if __name__ == "__main__":
    asyncio.run(main())