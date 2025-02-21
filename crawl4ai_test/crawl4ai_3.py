import asyncio
import nest_asyncio
from typing import List
from urllib.parse import urljoin, urlparse
import logging

from crawl4ai import AsyncWebCrawler, CacheMode, BrowserConfig, CrawlerRunConfig
from crawl4ai.content_filter_strategy import LLMContentFilter
from dotenv import load_dotenv
import os

load_dotenv()

url = "http://www.ditekjaya.co.id"
model_name=os.getenv("OPENAI_MODEL_NAME")
api_key=os.getenv("OPENAI_API_KEY")
base_url=os.getenv("OPENAI_BASE_URL")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebsiteCrawler:
    def __init__(
        self,
        base_url: str,
        model_name: str,
        api_key: str,
        api_base: str,
        instructions: str,
        max_pages: int = 50
    ):
        self.base_url = base_url
        self.domain = urlparse(base_url).netloc
        self.model_name = model_name
        self.api_key = api_key
        self.api_base = api_base
        self.instructions = instructions
        self.max_pages = max_pages
        self.visited_urls = set()
        self.markdown_contents = []

    def is_internal_url(self, url: str) -> bool:
        """Check if URL belongs to the same domain."""
        return self.domain in url

    async def process_page(self, url: str, crawler: AsyncWebCrawler, run_config: CrawlerRunConfig) -> List[str]:
        """Process a single page and return discovered URLs."""
        if url in self.visited_urls or len(self.visited_urls) >= self.max_pages:
            return []

        logger.info(f"Processing page: {url}")
        self.visited_urls.add(url)

        try:
            # Crawl the page
            result = await crawler.arun(url, config=run_config)
            html = result.cleaned_html

            # Extract links from the page
            new_urls = result.links
            internal_urls = [
                urljoin(self.base_url, url) 
                for url in new_urls 
                if self.is_internal_url(urljoin(self.base_url, url))
            ]

            # Process content with LLM
            filter = LLMContentFilter(
                provider=self.model_name,
                api_token=self.api_key,
                api_base=self.api_base,
                chunk_token_threshold=2 ** 12 * 2,  # 2048 * 2
                instruction=self.instructions,
                verbose=True
            )

            filtered_content = filter.filter_content(html, ignore_cache=True)
            if filtered_content:
                self.markdown_contents.extend(filtered_content)
                
            # Show token usage for this page
            filter.show_usage()
            
            return internal_urls

        except Exception as e:
            logger.error(f"Error processing {url}: {str(e)}")
            return []

    async def crawl(self):
        """Crawl the website and process all internal pages."""
        browser_config = BrowserConfig(
            headless=True,
            verbose=True
        )
        run_config = CrawlerRunConfig(cache_mode=CacheMode.ENABLED)

        async with AsyncWebCrawler(config=browser_config) as crawler:
            urls_to_process = [self.base_url]
            
            while urls_to_process and len(self.visited_urls) < self.max_pages:
                current_url = urls_to_process.pop(0)
                new_urls = await self.process_page(current_url, crawler, run_config)
                
                # Add new URLs to processing queue if they haven't been visited
                urls_to_process.extend([
                    url for url in new_urls 
                    if url not in self.visited_urls 
                    and url not in urls_to_process
                ])

            # Save all markdown content to file
            combined_markdown = "\n\n---\n\n".join(self.markdown_contents)
            with open("website_content.md", "w", encoding="utf-8") as f:
                f.write(combined_markdown)

            logger.info(f"Crawling completed. Processed {len(self.visited_urls)} pages.")
            return self.markdown_contents

async def main():
    # Load environment variables
    load_dotenv()
    
    url = "http://www.ditekjaya.co.id"
    model_name=os.getenv("OPENAI_MODEL_NAME")
    api_key=os.getenv("OPENAI_API_KEY")
    base_url=os.getenv("OPENAI_BASE_URL")
    
    instructions = """
    Convert all contents to English and extract meaningful information including:
    - Main content
    - Important details
    - Contact information
    - Products or services
    Format the output as clean markdown with proper sections.
    """

    # Initialize and run crawler
    crawler = WebsiteCrawler(
        base_url=url,
        model_name=model_name,
        api_key=api_key,
        api_base=base_url,
        instructions=instructions,
        max_pages=50  # Limit the number of pages to crawl
    )
    
    nest_asyncio.apply()
    await crawler.crawl()

if __name__ == "__main__":
    asyncio.run(main())