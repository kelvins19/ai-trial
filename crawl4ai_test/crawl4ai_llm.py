from crawl4ai.extraction_strategy import LLMExtractionStrategy
from pydantic import BaseModel, Field
import os, json
from crawl4ai import AsyncWebCrawler, CacheMode, BrowserConfig, CrawlerRunConfig, CacheMode
from dotenv import load_dotenv

load_dotenv()

model_name=os.getenv("OPENAI_MODEL_NAME")
api_key=os.getenv("OPENAI_API_KEY")
base_url=os.getenv("OPENAI_BASE_URL")

class OpenAIModelFee(BaseModel):
    model_name: str = Field(..., description="Name of the OpenAI model.")
    input_fee: str = Field(..., description="Fee for input token for the OpenAI model.")
    output_fee: str = Field(
        ..., description="Fee for output token for the OpenAI model."
    )

async def extract_structured_data_using_llm(provider: str, api_token: str = None, extra_headers: dict = None):
    print(f"\n--- Extracting Structured Data with {provider} ---")

    # Skip if API token is missing (for providers that require it)
    if api_token is None and provider != "ollama":
        print(f"API token is required for {provider}. Skipping this example.")
        return

    extra_args = {"extra_headers": extra_headers} if extra_headers else {}

    async with AsyncWebCrawler(verbose=True) as crawler:
        result = await crawler.arun(
            url="https://openai.com/api/pricing/",
            word_count_threshold=1,
            extraction_strategy=LLMExtractionStrategy(
                provider=provider,
                api_token=api_token,
                schema=OpenAIModelFee.schema(),
                extraction_type="schema",
                instruction="""Extract all model names along with fees for input and output tokens."
                "{model_name: 'GPT-4', input_fee: 'US$10.00 / 1M tokens', output_fee: 'US$30.00 / 1M tokens'}.""",
                **extra_args
            ),
            cach_mode = CacheMode.ENABLED
        )
        print(json.loads(result.extracted_content)[:5])

# Usage:
# await extract_structured_data_using_llm("huggingface/meta-llama/Meta-Llama-3.1-8B-Instruct", os.getenv("HUGGINGFACE_API_KEY"))
# await extract_structured_data_using_llm("ollama/llama3.2")
# await extract_structured_data_using_llm(model_name, api_key)