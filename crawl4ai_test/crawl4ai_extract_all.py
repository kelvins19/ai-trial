import asyncio
import os
import json
from collections import deque
from typing import Dict, List, Set
from crawl4ai.models import CrawlResult
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode, LXMLWebScrapingStrategy
from urllib.parse import urljoin, urlparse, urlunparse
from crawl4ai.content_filter_strategy import PruningContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from db.pinecone_db import store_embeddings_in_pinecone, search_pinecone
import aiohttp
import logging
import re

class WebScraper:
    def __init__(self, crawler: AsyncWebCrawler):
        self.crawler = crawler
        self.base_url = None
        self.last_modified_data = {}

    def is_valid_internal_link(self, link: str) -> bool:
        if not link or link.startswith('#'):
            return False
        
        parsed_base = urlparse(self.base_url)
        parsed_link = urlparse(link)
        
        return (parsed_base.netloc == parsed_link.netloc and
                parsed_link.path not in ['', '/'] and
                parsed_link.path.startswith(parsed_base.path))

    def normalize_url(self, url: str) -> str:
        parsed = urlparse(url)
        # Remove any fragments
        parsed = parsed._replace(fragment='')
        # Ensure the path doesn't end with a slash unless it's the root
        if parsed.path.endswith('/') and len(parsed.path) > 1:
            parsed = parsed._replace(path=parsed.path.rstrip('/'))
        return urlunparse(parsed)

    def join_url(self, base: str, url: str) -> str:
        joined = urljoin(base, url)
        parsed_base = urlparse(self.base_url)
        parsed_joined = urlparse(joined)
        
        # Ensure the joined URL starts with the base path
        if not parsed_joined.path.startswith(parsed_base.path):
            # If it doesn't, prepend the base path
            new_path = parsed_base.path.rstrip('/') + '/' + parsed_joined.path.lstrip('/')
            parsed_joined = parsed_joined._replace(path=new_path)
        
        return urlunparse(parsed_joined)

    async def get_last_modified(self, url: str) -> str:
        async with aiohttp.ClientSession() as session:
            async with session.head(url) as response:
                last_modified = response.headers.get('Last-Modified')
                if last_modified:
                    print(f"URL {url} Last Modified: {last_modified}")
                    return last_modified
                
                # If Last-Modified header is not available, get the page content
                async with session.get(url) as response:
                    content = await response.text()
                    match = re.search(r'"dateModified":"([^"]+)"', content)
                    if match:
                        print(f"URL {url} Last Modified: {match.group(1)}")
                        return match.group(1)
                    
                    print(f"URL {url} Last Modified: {match}")
                return None

    async def scrape(self, start_url: str, max_depth: int) -> Dict[str, CrawlResult]:
        self.base_url = start_url
        results: Dict[str, CrawlResult] = {}
        queue: deque = deque([(self.base_url, 0)])
        visited: Set[str] = set()

        config = CrawlerRunConfig(
            cache_mode=CacheMode.ENABLED,
            excluded_tags=['nav', 'footer', 'aside'],
            remove_overlay_elements=True,
            # scraping_strategy=LXMLWebScrapingStrategy()  # Faster alternative to default BeautifulSoup
        )

        while queue:
            current_url, current_depth = queue.popleft()
            is_current_url_changed = True
            if current_url in visited or current_depth > max_depth:
                continue
            
            visited.add(current_url)
            
            last_modified = await self.get_last_modified(current_url)
            if last_modified and self.last_modified_data.get(current_url) == last_modified:
                print(f"URL {current_url} has not been modified since the last crawl.")
                # Continue to next URL in the queue
                is_current_url_changed = False

            if is_current_url_changed:
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
                    # scraping_strategy=LXMLWebScrapingStrategy()  # Faster alternative to default BeautifulSoup
                )

            result = await self.crawler.arun(url=current_url, config=config)
            
            if result.success:
                results[current_url] = None
                if is_current_url_changed:
                    results[current_url] = result
                self.last_modified_data[current_url] = last_modified
                
                if current_depth < max_depth:
                    internal_links = result.links.get('internal', [])
                    for link in internal_links:
                        full_url = self.join_url(current_url, link['href'])
                        if self.is_valid_internal_link(full_url) and full_url not in visited:
                            queue.append((full_url, current_depth + 1))

        return results

    def save_results_to_markdown(self, results: Dict[str, CrawlResult], folder_name: str):
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        
        markdown_data = {}
        for url, result in results.items():
            if result != None:
                parsed_url = urlparse(url)
                path = parsed_url.path.strip('/')
                if not path:
                    path = 'index'
                file_path = os.path.join(folder_name, f"{path}.md")
                file_dir = os.path.dirname(file_path)
                if not os.path.exists(file_dir):
                    os.makedirs(file_dir)
                with open(file_path, 'w') as file:
                    file.write(f"# {url}\n\n")
                    file.write(f"Last Modified: {self.last_modified_data[url]}\n\n")
                    file.write(f"## Content\n\n")
                    markdown_content = result.markdown_v2.fit_markdown or "No content available."
                    file.write(markdown_content)
                    file.write(f"\n\n## Images\n")
                    for img in result.media.get('images', []):
                        file.write(f"- Image URL: {img['src']}, Alt: {img['alt']}\n")
                    markdown_data[url] = markdown_content
        
        # Store the markdown content as embeddings in Pinecone
        store_embeddings_in_pinecone(folder_name, markdown_data, chunk_size=100)

    def load_last_modified_data(self, file_path: str):
        if (os.path.exists(file_path)):
            with open(file_path, 'r') as file:
                self.last_modified_data = json.load(file)

    def save_last_modified_data(self, file_path: str):
        with open(file_path, 'w') as file:
            json.dump(self.last_modified_data, file)

async def main(start_url: str, depth: int):
    async with AsyncWebCrawler() as crawler:
        scraper = WebScraper(crawler)
        last_modified_file = 'last_modified.json'
        scraper.load_last_modified_data(last_modified_file)
        results = await scraper.scrape(start_url, depth)
        
        folder_name = urlparse(start_url).netloc.replace('.', '-').lower()
        scraper.save_results_to_markdown(results, folder_name)
        scraper.save_last_modified_data(last_modified_file)

    number_of_crawled_pages = 0
    for url, result in results.items():
        if result != None:
            number_of_crawled_pages += 1

    print(f"Content changes on {number_of_crawled_pages} pages")
    

if __name__ == "__main__":
    start_url = "http://www.ditekjaya.co.id"
    depth = 10
    asyncio.run(main(start_url, depth)) 

    # folder_name = "www-ditekjaya-co-id"
    # # Example search query
    # query = "tracera"
    # print(f"Folder Name {folder_name}")
    # search_results = search_pinecone(folder_name, query)
    # print(f"Search results for query '{query}':")
    # if search_results:
    #     for match in search_results.get('matches', []):
    #         print(f"- {match['id']}: {match['score']}")
    # else:
    #     print("No results found.")