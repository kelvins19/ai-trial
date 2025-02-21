import asyncio
import nest_asyncio
nest_asyncio.apply()

from crawl4ai import AsyncWebCrawler, CacheMode, BrowserConfig, CrawlerRunConfig

# Ensure Playwright browsers are installed
import subprocess
subprocess.run(["playwright", "install"], check=True)

url = "http://www.ditekjaya.co.id"
url = "http://www.ditekjaya.co.id/produk-kami/chromatography-systems/gas-chromatographs/tracera/"

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
        print(result.markdown_v2.fit_markdown)

asyncio.run(clean_content())

async def link_analysis():
    async with AsyncWebCrawler() as crawler:
        config = CrawlerRunConfig(
            cache_mode=CacheMode.ENABLED,
            exclude_external_links=True,
            exclude_social_media_links=True,
            # exclude_domains=["facebook.com", "twitter.com"]
        )
        result = await crawler.arun(
            url=url,
            config=config,
        )
        print(f"Found {len(result.links['internal'])} internal links")
        print(f"Found {len(result.links['external'])} external links")

        for link in result.links['internal'][:5]:
            print(f"Href: {link['href']}\nText: {link['text']}\n")


# asyncio.run(link_analysis())

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