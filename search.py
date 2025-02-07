import yaml
import os
import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from urllib.parse import urlparse
from serpapi import GoogleSearch
from dotenv import load_dotenv
from openai import OpenAI
import json
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import aiohttp
import asyncio
import ssl
import numpy as np
import openai

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
}

# Create a session with retry strategy
def create_session():
    session = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[500, 502, 503, 504]
    )
    adapter = HTTPAdapter(
        pool_connections=20,
        pool_maxsize=20,
        max_retries=retries
    )
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

cwd = os.getcwd()
CONFIG_DIRS = os.path.join(cwd, 'conf')
DATA_DIRS = os.path.join(cwd, 'data')
load_dotenv()
SERP_API_KEY = os.getenv("SERP_API_KEY")
openai.api_key = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


class webSearch(object):
    def __init__(self):
        self.use_urls = {
            'pubmed': ["pmc.ncbi.nlm.nih.gov", "pubmed.ncbi.nlm.nih.gov", "www.ncbi.nlm.nih.gov"]
        }
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=GEMINI_API_KEY,
        )
        self.session = create_session()
        # Increase workers for parallel processing
        self.content_executor = ThreadPoolExecutor(max_workers=20)
        # Add search executor for parallel API calls
        self.search_executor = ThreadPoolExecutor(max_workers=5)
        # Add summarization executor
        self.summary_executor = ThreadPoolExecutor(max_workers=10)
        # Cache for summaries
        self.summary_cache = {}
        # Shorter prompt for faster processing
        self.summary_prompt = '''As a clinical pharmacist, extract and concisely summarize only clinically relevant information from this text. 

                                    Format: Brief bullet points. Skip categories with no relevant information.

                                    Input text: "{}"
                                    '''

    async def fetch_content(self, session, url, start_time):
        fetch_start = time.time()
        if any(ext in url.lower() for ext in ['.pdf', '.doc', '.docx', '.ppt', '.pptx', '.txt']):
            return None
            
        try:
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            async with session.get(url, headers=headers, timeout=5, ssl=ssl_context) as response:
                if response.status != 200:
                    return None
                    
                content_type = response.headers.get('Content-Type', '').lower()
                if not ('text/html' in content_type or 'application/xhtml' in content_type):
                    return None
                
                try:
                    text = await response.text(encoding='utf-8')
                except UnicodeDecodeError:
                    try:
                        text = await response.text()
                    except:
                        return None
                fetch_end = time.time()
                with open('websearch_timing.txt', 'a', encoding='utf-8') as f:
                    f.write(f"Fetched {url}: {fetch_end - fetch_start:.3f} seconds\n")
                return text
        except Exception as e:
            print(f"Error fetching {url}: {e}")
            return None

    async def process_single_result(self, session, result, start_time):
        process_start = time.time()
        try:
            url = result.get("link")
            if not url:
                return None
            
            content = await self.fetch_content(session, url, start_time)
            if not content:
                return None

            parse_start = time.time()
            soup = BeautifulSoup(content, 'html.parser')
            for tag in soup(['script', 'style', 'nav', 'footer', 'header']):
                tag.decompose()
            
            main_content = soup.find('main') or soup.find('article') or soup.find('div', {'class': ['content', 'main']})
            if main_content:
                content = " ".join(p.get_text(strip=True) for p in main_content.find_all('p'))
            else:
                content = " ".join(p.get_text(strip=True) for p in soup.find_all('p', recursive=False))
            
            parse_end = time.time()
            with open('websearch_timing.txt', 'a', encoding='utf-8') as f:
                f.write(f"Parsed {url}: {parse_end - parse_start:.3f} seconds\n")
            
            if len(content) < 100:
                return None

            return {
                'url': url,
                'title': result.get('title', ''),
                'content': content,
                'snippet': result.get('snippet', '')
            }
        except Exception as e:
            print(f"Error processing result: {e}")
            return None
        finally:
            process_end = time.time()
            with open('websearch_timing.txt', 'a', encoding='utf-8') as f:
                f.write(f"Total processing for URL: {process_end - process_start:.3f} seconds\n")

    async def process_results_async(self, results):
        process_start = time.time()
        connector = aiohttp.TCPConnector(ssl=False)
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = [self.process_single_result(session, result, process_start) for result in results]
            processed_results = await asyncio.gather(*tasks, return_exceptions=True)
            valid_results = [r for r in processed_results if r is not None and not isinstance(r, Exception)]
            
        process_end = time.time()
        with open('websearch_timing.txt', 'a', encoding='utf-8') as f:
            f.write(f"Total async processing: {process_end - process_start:.3f} seconds\n")
        return valid_results

    def search_api(self, queries):
        search_start = time.time()
        try:
            # Batch all queries into a single API call with OR operator
            combined_query = " OR ".join(f"({q})" for q in queries)
            params = {
                "engine": "google",
                "q": combined_query,
                "api_key": SERP_API_KEY,
                "num": 20  # Increased to account for combined queries
            }
            search = GoogleSearch(params)
            results = search.get_dict()
            search_end = time.time()
            with open('websearch_timing.txt', 'a', encoding='utf-8') as f:
                f.write(f"Combined SERP API search: {search_end - search_start:.3f} seconds\n")
            return results.get('organic_results', [])
        except Exception as e:
            print(f"Error in search_api: {e}")
            return []

    def run(self, queries):
        with open('websearch_timing.txt', 'w', encoding='utf-8') as f:
            f.write("=== Web Search Timing Log ===\n\n")
        
        total_start = time.time()
        
        # Get all results in one API call
        results = self.search_api(queries)
        
        # Process all results in parallel
        processed_results = asyncio.run(self.process_results_async(results))
        
        # Deduplicate results
        final_results = []
        seen_urls = set()
        for result in processed_results:
            if result and result['url'] not in seen_urls:
                final_results.append(result)
                seen_urls.add(result['url'])

        total_end = time.time()
        with open('websearch_timing.txt', 'a', encoding='utf-8') as f:
            f.write(f"\nTotal web search time: {total_end - total_start:.3f} seconds\n")
            f.write(f"Number of results: {len(final_results)}\n")
            f.write("===========================\n")

        with open('final-results.txt', 'w', encoding='utf-8') as f:
            json.dump(final_results, f, ensure_ascii=False, indent=4)

        return final_results

    def summarize_single(self, content):
        """Summarize a single piece of content"""
        try:
            # Skip if content is too short
            if len(content) < 100:
                return None
                
            if len(content) > 5000:
                content = content[:5000]
        
            
            # Make API call with shorter prompt
            response = self.client.chat.completions.create(
                model='google/gemini-flash-1.5',
                messages=[{'role': 'user', 'content': self.summary_prompt.format(content)}]
            )
            summary = response.choices[0].message.content
            
            return summary
        except Exception as e:
            print(f"Error summarizing content: {e}")
            return None

    def summarize_scraped_content(self, content):
        """Main summarization function"""
        try:
            summary = self.summarize_single(content)
            # if summary:
                # Write summaries in batches to reduce I/O
                # with open('summarized-scraped-content.txt', 'a', encoding='utf-8') as f:
                #     f.write(summary + "\n")
            return summary
        except Exception as e:
            print(f"Error in summarize_scraped_content: {e}")
            return None