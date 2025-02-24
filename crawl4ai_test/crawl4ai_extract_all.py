import asyncio
import os
from collections import deque
from typing import Dict, List, Set
from crawl4ai.models import CrawlResult
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode
from urllib.parse import urljoin, urlparse, urlunparse
from crawl4ai.content_filter_strategy import PruningContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from db.pinecone_db import store_embeddings_in_pinecone, search_pinecone

class WebScraper:
    def __init__(self, crawler: AsyncWebCrawler):
        self.crawler = crawler
        self.base_url = None

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

    async def scrape(self, start_url: str, max_depth: int) -> Dict[str, CrawlResult]:
        self.base_url = start_url
        results: Dict[str, CrawlResult] = {}
        queue: deque = deque([(self.base_url, 0)])
        visited: Set[str] = set()

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

        while queue:
            current_url, current_depth = queue.popleft()
            if current_url in visited or current_depth > max_depth:
                continue
            
            visited.add(current_url)
            
            result = await self.crawler.arun(url=current_url, config=config)
            
            if result.success:
                results[current_url] = result
                
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
                file.write(f"## Content\n\n")
                markdown_content = result.markdown_v2.fit_markdown or "No content available."
                file.write(markdown_content)
                file.write(f"\n\n## Images\n")
                for img in result.media.get('images', []):
                    file.write(f"- Image URL: {img['src']}, Alt: {img['alt']}\n")
                markdown_data[url] = markdown_content
        
        # Store the markdown content as embeddings in Pinecone
        store_embeddings_in_pinecone(folder_name, markdown_data)

async def main(start_url: str, depth: int):
    async with AsyncWebCrawler() as crawler:
        scraper = WebScraper(crawler)
        results = await scraper.scrape(start_url, depth)
        
        folder_name = urlparse(start_url).netloc.replace('.', '-').lower()
        scraper.save_results_to_markdown(results, folder_name)
    
    print(f"Crawled {len(results)} pages:")
    for url, result in results.items():
        print(f"- {url}: {len(result.links.get('internal', []))} internal links, {len(result.links.get('external', []))} external links")
    

if __name__ == "__main__":
    start_url = "http://www.ditekjaya.co.id"
    depth = 10
    asyncio.run(main(start_url, depth))

    folder_name = "www-ditekjaya-co-id"
    # Example search query
    query = "tracera"
    print(f"Folder Name {folder_name}")
    search_results = search_pinecone(folder_name, query)
    print(f"Search results for query '{query}':")
    if search_results:
        for match in search_results.get('matches', []):
            print(f"- {match['id']}: {match['score']}")
    else:
        print("No results found.")