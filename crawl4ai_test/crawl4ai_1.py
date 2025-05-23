import asyncio
import nest_asyncio
nest_asyncio.apply()

from crawl4ai import AsyncWebCrawler, CacheMode, BrowserConfig, CrawlerRunConfig, CacheMode, LLMContentFilter
from crawl4ai.extraction_strategy import JsonCssExtractionStrategy, JsonXPathExtractionStrategy

# Ensure Playwright browsers are installed
import subprocess
subprocess.run(["playwright", "install"], check=True)

# url = "http://www.ditekjaya.co.id"
# url = "http://www.ditekjaya.co.id/produk-kami/chromatography-systems/gas-chromatographs/tracera/"
url = "https://i12katong.com.sg/whatsOn"

from dotenv import load_dotenv
import os

load_dotenv()
model_name=os.getenv("OPENAI_MODEL_NAME")
api_key=os.getenv("OPENAI_API_KEY")
base_url=os.getenv("OPENAI_BASE_URL")

async def simple_crawl():
    crawler_run_config = CrawlerRunConfig( cache_mode=CacheMode.BYPASS)
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(
            url=url,
            config=crawler_run_config
        )
        print(result.markdown_v2.raw_markdown[:500].replace("\n", " -- "))  # Print the first 500 characters

# asyncio.run(simple_crawl())

from crawl4ai.content_filter_strategy import PruningContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

async def clean_content():
    async with AsyncWebCrawler(verbose=True) as crawler:
        config = CrawlerRunConfig(
            cache_mode=CacheMode.ENABLED,
            excluded_tags=['nav', 'footer', 'aside'],
            remove_overlay_elements=True,
            markdown_generator=DefaultMarkdownGenerator(
                content_filter=PruningContentFilter(threshold=0.48, threshold_type="fixed", min_word_threshold=0),
                options={
                    "ignore_links": True
                }
            ),
        )
        result = await crawler.arun(
            url=url,
            config=config,
        )
        full_markdown_length = len(result.markdown_v2.raw_markdown)
        fit_markdown_length = len(result.markdown_v2.fit_markdown)
        print(f"Full Markdown Length: {full_markdown_length}")
        print(f"Fit Markdown Length: {fit_markdown_length}")
        print(f"Scrapped Content for URL {url}")
        print(result.markdown_v2.fit_markdown)

# asyncio.run(clean_content())

async def link_analysis():
    async with AsyncWebCrawler() as crawler:
        # 1. Define a simple extraction schema
        schema = {
            "name": "NEXT DATA",
            "baseSelector": "//*[@id=\"__NEXT_DATA__\"]", 
            "fields": [
                {
                    "name": "data",
                    "selector": ".//*[@id=\"__NEXT_DATA__\"]/text()",
                    "type": "text"
                }
            ]
        }
        # 2. Create the extraction strategy
        extraction_strategy = JsonXPathExtractionStrategy(schema, verbose=True)

        config = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            exclude_external_links=False,
            exclude_social_media_links=True,
            extraction_strategy=extraction_strategy
            # exclude_domains=["facebook.com", "twitter.com"]
        )
        result = await crawler.arun(
            url=url,
            config=config,
        )
        print(f"Found {len(result.links['internal'])} internal links")
        print(f"Found {len(result.links['external'])} external links")

        for link in result.links['external'][:5]:
            print(f"External Href: {link['href']}\nText: {link['text']}\n")

        for link in result.links['internal'][:5]:
            print(f"Internal Href: {link['href']}\nText: {link['text']}\n")

        # Extract API call links
        api_calls = [link for link in result.links['internal'] if 'cms' in link['href']]
        print(f"Found {len(api_calls)} API call links")
        for api_call in api_calls:
            print(f"API Call Href: {api_call['href']}\nText: {api_call['text']}\n")

        print(f"Extracted content: {result.extracted_content}")
        print(f"Metadata: {result.metadata}")
        # print(f"Cleaned HTML: {result.cleaned_html}")
        # print(f"Fit HTML: {result.fit_html}")
        # print(f"HTML: {result.html}")

asyncio.run(link_analysis())

async def media_handling():
    async with AsyncWebCrawler() as crawler:
        config = CrawlerRunConfig(
            cache_mode=CacheMode.ENABLED,
            exclude_external_images=False,
            # screenshot=True # Set this to True if you want to take a screenshot
        )
        result = await crawler.arun(
            url=url,
            config=config,
        )
        for img in result.media['images'][:5]:
            print(f"Image URL: {img['src']}, Alt: {img['alt']}, Score: {img['score']}")

# asyncio.run(media_handling())


async def ai_content_search():
    async with AsyncWebCrawler() as crawler:
        config = CrawlerRunConfig(
            cache_mode=CacheMode.ENABLED,
            ai_content_search=LLMExtractionStrategy(
                model=model_name,
                base_url=base_url,
                api_key=api_key,
                search_query="how many products does this website have?"
            )
        )
        result = await crawler.arun(
            url=url,
            config=config,
        )
        print(f"AI Search Results: {result.ai_content_search_results}")

# asyncio.run(ai_content_search())

async def llm_extract_products():
    instructions = """
                Extract all available products from the website, including:
                - Product names
                - Description details
                - Images
                Format the output as clean markdown with proper sections for each product.
            """
    
    instructions = """
            Convert all contents to english. Then
            Extract all representative offices of ditekjaya, including all of its details.
            Format the output as clean markdown with proper sections for each office.
        """
    browser_config = BrowserConfig(
        headless=True,
        verbose=True
    )
    run_config = CrawlerRunConfig(cache_mode=CacheMode.ENABLED)


    async with AsyncWebCrawler(config=browser_config) as crawler:
        result = await crawler.arun(url, config=run_config)
        html = result.cleaned_html

        filter = LLMContentFilter(
            provider=model_name,
            api_token=api_key,
            api_base=base_url,
            chunk_token_threshold=2 ** 12 * 2, # 2048 * 2
            instruction=instructions,
            verbose=True
        )

        #  Apply filtering
        filtered_content = filter.filter_content(html, ignore_cache = True)
        
        # Show results
        print("\nFiltered Content Length:", len(filtered_content))
        print("\nFirst 500 chars of filtered content:")
        if filtered_content:
            print(filtered_content[0][:500])
        
        # Save on disc the markdown version
        with open("filtered_content.md", "w", encoding="utf-8") as f:
            f.write("\n".join(filtered_content))
        
        # Show token usage
        filter.show_usage()

# asyncio.run(llm_extract_products())