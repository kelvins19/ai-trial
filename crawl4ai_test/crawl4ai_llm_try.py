import os
import asyncio
from crawl4ai.extraction_strategy import LLMExtractionStrategy
from pydantic import BaseModel, Field
import os, json
from crawl4ai import AsyncWebCrawler, CacheMode, BrowserConfig, CrawlerRunConfig, CacheMode
from dotenv import load_dotenv

load_dotenv()


async def determine_relevant_pages(url: str, provider: str, api_token: str = None, extra_headers: dict = None):
    print(f"\n--- Determining Relevant Pages with {provider} ---")

    extra_args = {"extra_headers": extra_headers} if extra_headers else {}

    async with AsyncWebCrawler(verbose=True) as crawler:
        result = await crawler.arun(
            url=url,
            word_count_threshold=1,
            extraction_strategy=LLMExtractionStrategy(
                provider=provider,
                api_token=api_token,
                api_base=os.getenv("OPENAI_BASE_URL"),
                schema=None,
                extraction_type="text",
                instruction="""Analyze the content of the main page and identify links that are relevant for extraction. 
                Provide a list of URLs that should be further crawled based on their relevance to the main content.""",
                **extra_args
            ),
            cache_mode=CacheMode.ENABLED
        )
        print("Extracted Content:", result.extracted_content)  # Debug statement
        try:
            relevant_links = json.loads(result.extracted_content)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            relevant_links = []
        return relevant_links

async def test_determine_relevant_pages():
    url = "https://i12katong.com.sg"
    provider = os.getenv("OPENAI_MODEL_NAME")
    api_token = os.getenv("OPENAI_API_KEY")

    relevant_pages = await determine_relevant_pages(url, provider, api_token)
    print("Relevant Pages:")
    for page in relevant_pages:
        print(page)

if __name__ == "__main__":
    asyncio.run(test_determine_relevant_pages())
