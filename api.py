import subprocess
from flask import Flask, request, jsonify, Response
from pydantic import BaseModel, Field
import json
from search import webSearch
import datetime as dt
import time
import json
from flask_cors import CORS
import tiktoken
import os
import numpy as np
from collections import defaultdict
import math
import faiss
import logging
from werkzeug.serving import WSGIRequestHandler
from dotenv import load_dotenv
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import openai

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("GEMINI_API_KEY")
)


# Configure logging to ignore socket.io requests
class SocketIOFilter(logging.Filter):
    def filter(self, record):
        return 'socket.io' not in record.getMessage()


# Apply filter to Werkzeug logger
logging.getLogger('werkzeug').addFilter(SocketIOFilter())

openai.api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)
# model = SentenceTransformer('all-MiniLM-L6-v2')
websearch = webSearch()
CORS(app)


class References(BaseModel):
    title: str = Field(description="Title of Reference")
    url: str = Field(description="Link/url to reference")
    citation_number: str = Field(description="Index of the citation")


class SearchResponse(BaseModel):
    response: str = Field(description="response content")
    citations: list[References] = Field(
        description="List of citations referenced in response")


class Queries(BaseModel):
    query: list[str] = Field(description="List of queries to search")


class TFIDFSearcher:
    def __init__(self):
        self.document_freq = defaultdict(int)
        self.vocab = {}
        self.idf = {}
        self.index = None
        self.doc_map = []

    def _calculate_tf(self, text):
        words = text.lower().split()
        word_count = defaultdict(int)
        for word in words:
            word_count[word] += 1
        return word_count

    def fit(self, documents):
        for doc in documents:
            word_counts = self._calculate_tf(doc)
            for word in word_counts:
                if word not in self.vocab:
                    self.vocab[word] = len(self.vocab)
                self.document_freq[word] += 1

        num_docs = len(documents)
        self.idf = {word: math.log(num_docs / freq)
                    for word, freq in self.document_freq.items()}

        vectors = np.zeros((len(documents), len(self.vocab)), dtype=np.float32)
        for i, doc in enumerate(documents):
            vectors[i] = self._text_to_vector(doc)

        self.index = faiss.IndexFlatIP(len(self.vocab))
        self.index.add(vectors)
        return self

    def _text_to_vector(self, text):
        vector = np.zeros(len(self.vocab), dtype=np.float32)
        word_counts = self._calculate_tf(text)

        for word, count in word_counts.items():
            if word in self.vocab:
                idx = self.vocab[word]
                tf = 1 + math.log(count)
                vector[idx] = tf * self.idf.get(word, 0)

        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm
        return vector

    def search(self, query, k=5):
        query_vector = self._text_to_vector(query)
        query_vector = query_vector.reshape(1, -1)

        # Search using FAISS
        similarities, indices = self.index.search(query_vector, k)
        return similarities[0], indices[0]


def rank_results_with_tfidf(query, all_query_results):
    contents = []
    content_to_result_map = {}

    # Extract contents and build mapping
    for query_result in all_query_results:
        content = query_result.get('content', '').strip()
        if content:
            contents.append(content)
            content_to_result_map[len(contents) - 1] = query_result

    if not contents:
        print("No valid contents found for TFIDF ranking")
        return []

    # Initialize and fit TFIDF searcher
    try:
        searcher = TFIDFSearcher()
        searcher.fit(contents)

        # Get top matches using TFIDF
        k = min(10, len(contents))  # Don't request more items than we have
        if k == 0:
            print("No contents to rank")
            return []

        similarities, indices = searcher.search(query, k=k)

        # Format results, filtering out invalid indices
        top_results = []
        for sim, idx in zip(similarities, indices):
            if isinstance(idx, (int, np.integer)):  # Convert numpy types to int
                idx = int(idx)

            if idx >= 0 and idx < len(contents) and idx in content_to_result_map:
                result = content_to_result_map[idx]
                top_results.append({
                    'url': result['url'],
                    'title': result.get('title', ''),
                    'result': {
                        'content': result['content'],
                        'snippet': result.get('snippet', '')
                    }
                })
        # with open('top_results.json', 'w', encoding='utf-8') as f:
        #     f.write(f'Number of results: {len(top_results)}\n')
        #     json.dump(top_results, f, indent=4)
        return top_results

    except Exception as e:
        print(f"Error in TFIDF ranking: {e}")
        # Fall back to returning the first few results without ranking
        fallback_results = []
        for idx, result in content_to_result_map.items():
            if len(fallback_results) >= 10:
                break
            fallback_results.append({
                'url': result['url'],
                'title': result.get('title', ''),
                'result': {
                    'content': result['content'],
                    'snippet': result.get('snippet', '')
                }
            })
        return fallback_results




def queryExpansion(initial_query):
    queryExpansionPrompt = f"""
            Generate 5 search queries optimized for web search engines from this clinical question.
            Make each query clear, specific, and likely to return relevant medical information.

            Initial query: "{initial_query}"

            Return ONLY a list of 5 queries formatted exactly like this:
            [
                "web optimized query 1",
                "web optimized query 2",
                "web optimized query 3",
                "web optimized query 4",
                "web optimized query 5"
            ]
        """

    response = client.chat.completions.create(
        model='google/gemini-flash-1.5',
        messages=[{'role': 'user', 'content': queryExpansionPrompt}],
        stream=True
    )

    # Handle streaming response
    full_response = ""
    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            full_response += chunk.choices[0].delta.content

    try:
        list_queries = json.loads(full_response)
    except json.JSONDecodeError:
        # If JSON parsing fails, extract the list from the string response
        import re
        matches = re.findall(r'"([^"]*)"', full_response)
        list_queries = matches[:5]  # Take only first 5 matches

    with open('expanded_queries.json', 'w', encoding='utf-8') as f:
        json.dump(list_queries, f)

    return list_queries


def chunk_text(text, max_length=512):
    words = text.split()
    return [' '.join(words[i:i + max_length]) for i in range(0, len(words), max_length)]


def token_reduce(text):
    encoding = tiktoken.get_encoding("o200k_base")
    tokens = len(encoding.encode(text))
    if tokens > 100000:
        tokenized_content = encoding.encode(text)
        truncated_tokenized_content = tokenized_content[:100000]
        truncated_content = encoding.decode(truncated_tokenized_content)
        text = truncated_content        
    with open('prompt.txt', 'w', encoding='utf-8') as f:
        f.write(f'Token count: {tokens}\n\n\n')
        f.write(text)

    return text


def summarizer(query, sources):
    # Format context from structured sources
    context = '\n\n'.join([f"Source {i+1}:\nTitle: {source['title']}\nContent: {source['content']}" 
                           for i, source in enumerate(sources)])
    context = token_reduce(context)
    
    # Create a citations map for reference
    citations = {str(i+1): {'title': source['title'], 'url': source['url']}
                for i, source in enumerate(sources)}
    
    # Create a formatted list of available sources with URLs
    available_sources = '\n'.join([f"[{i+1}] {source['title']} - {source['url']}" 
                                  for i, source in enumerate(sources)])
    
    prompt_content = (
        f"Analyze and summarize the medical literature about {query}. Format your response as a JSON object with the following structure:\n"
        f"{{\n"
        f"  'overview': 'Brief overview of the topic',\n"
        f"  'analysis': 'Detailed analysis with citations [N]',\n"
        f"  'recommendations': 'Clinical recommendations with citations [N]',\n"
        f"  'key_takeaways': ['Key point 1', 'Key point 2', ...],\n"
        f"  'citations': [{{ 'number': 'N', 'reference': 'Author/Source. Title.' }}]\n"
        f"}}\n\n"
        f"Use [N] for citations where N is the source number.\n\n"
        f"Available sources with URLs:\n"
        + available_sources
        + f"\n\nContext:\n{context}"
    )
    
    prompt = {
        "messages": [{
            "role": "user",
            "content": prompt_content
        }]
    }
    
    response = client.chat.completions.create(
        model='google/gemini-flash-1.5',
        messages=[{'role': 'user', 'content': prompt_content}]
    )
    
    try:
        # Try to parse as JSON
        content = response.text if hasattr(response, 'text') else response.choices[0].message.content
        
        # Remove markdown code block if present
        if content.startswith('```') and '```' in content[3:]:
            # Extract content between first and last ```
            content = content.split('```', 2)[1]
            # Remove language identifier if present
            if '\n' in content:
                content = content.split('\n', 1)[1]
        
        # Clean up potential formatting issues
        content = content.strip()
        if content.startswith('```json'):
            content = content.replace('```json', '', 1)
        elif content.startswith('```'):
            content = content.replace('```', '', 1)
        if content.endswith('```'):
            content = content[:-3]
        content = content.strip()
        
        # Fix the common issue with periods after quotes
        content = content.replace('",.',  '",')
        content = content.replace('".',  '",')
        
        # Handle any escaped quotes
        content = content.replace('\\"', '"')
        
        try:
            result = json.loads(content)
        except json.JSONDecodeError as e:
            print(f"Initial JSON parsing failed: {e}")
            # If it still fails, try one more time with more aggressive cleaning
            content = content.replace('".}', '"}')
            content = content.replace('".]', '"]')
            result = json.loads(content)
        
        # Update citations with URLs from our sources
        if 'citations' in result:
            for citation in result['citations']:
                citation_num = citation['number']
                if citation_num in citations:
                    citation['url'] = citations[citation_num]['url']
        return result
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON response: {e}\nResponse: {content}")
        # If JSON parsing fails, extract sections from text
        sections = content.split('\n\n')
        overview = ''
        analysis = ''
        recommendations = ''
        key_takeaways = []
        citations = []
        
        current_section = ''
        for section in sections:
            if section.startswith('Overview'):
                overview = section.replace('Overview', '', 1).strip()
            elif section.startswith('Analysis'):
                analysis = section.replace('Analysis', '', 1).strip()
            elif section.startswith('Recommendations'):
                recommendations = section.replace('Recommendations', '', 1).strip()
            elif section.startswith('Key Takeaways'):
                takeaways = section.replace('Key Takeaways', '', 1).strip().split('\n')
                key_takeaways = [t.strip('- ') for t in takeaways if t.strip()]
            elif section.startswith('Citations') or section.startswith('References'):
                refs = section.replace('Citations', '', 1).replace('References', '', 1).strip().split('\n')
                for ref in refs:
                    if ref.strip():
                        citations.append({'number': str(len(citations)+1), 'reference': ref.strip('- []')})
        
        return {
            'overview': overview,
            'analysis': analysis,
            'recommendations': recommendations,
            'key_takeaways': key_takeaways,
            'citations': citations
        }

def process_single_result(result, searcher):
    if 'result' in result and 'content' in result['result']:
        content = searcher.summarize_scraped_content(
            result['result']['content'])
        return {
            'content': content,
            'url': result['url'],
            'title': result['title']
        }
    return None


def process_results(top_results):
    results_array = []
    searcher = webSearch()

    with ThreadPoolExecutor(max_workers=10) as executor:
        # Submit all results for processing
        future_to_result = {executor.submit(process_single_result, result, searcher): result
                            for result in top_results}

        # Collect results as they complete
        for future in as_completed(future_to_result):
            try:
                processed = future.result()
                if processed:
                    results_array.append({
                        'title': processed['title'],
                        'url': processed['url'],
                        'content': processed['content']
                    })
            except Exception as e:
                print(f"Error processing result: {e}")
                continue

    return results_array


@app.route('/api/webSearch', methods=['POST'])
def sherlockSearch():
    try:
        data = request.get_json()
        initial_query = data.get('query', '')
        if not initial_query:
            return jsonify({"error": "No query provided"}), 400

        with open('summarized-scraped-content.txt', 'w', encoding='utf-8') as f:
            f.write('')
        with open('final-results.json', 'w', encoding='utf-8') as f:
            f.write('')
        with open('final-urls.txt', 'w', encoding='utf-8') as f:
            f.write('')

        overall_start = time.time()
        with open('timestamp.txt', 'w', encoding='utf-8') as f:
            f.write('Time records for Sherlock Search\n\n')

        # Stage 1: Query Expansion
        time_before = time.time()
        expanded_queries = queryExpansion(initial_query)
        time_after = time.time()
        time_taken = time_after - time_before
        with open('timestamp.txt', 'a', encoding='utf-8') as f:
            f.write("===== QUERY EXPANSION =====\n")
            f.write(f"Time taken for query expansion: {time_taken:.4f} seconds\n")
            f.write("===========================\n\n")

        # Stage 2: Web Search
        time_before = time.time()
        searcher = webSearch()
        outs = searcher.run(expanded_queries)
        time_after = time.time()
        time_taken = time_after - time_before
        with open('timestamp.txt', 'a', encoding='utf-8') as f:
            f.write("========== WEB SEARCH ==========\n")
            f.write(f"Time taken for web search: {time_taken:.4f} seconds\n")
            f.write("===============================\n\n")

        # Stage 3: Ranking Results
        time_before = time.time()
        top_results = rank_results_with_tfidf(initial_query, outs)
        time_after = time.time()
        time_taken = time_after - time_before
        with open('timestamp.txt', 'a', encoding='utf-8') as f:
            f.write("======= RANKING RESULTS =======\n")
            f.write(f"Time taken for ranking: {time_taken:.4f} seconds\n")
            f.write("===============================\n\n")

        if not top_results:
            return jsonify({"error": "Failed to rank results"}), 500

        # Stage 4: Process Results
        time_before = time.time()
        results = process_results(top_results)
        time_after = time.time()
        time_taken = time_after - time_before
        with open('timestamp.txt', 'a', encoding='utf-8') as f:
            f.write("======= PROCESS RESULTS =======\n")
            f.write(f"Time taken for processing results: {
                    time_taken:.4f} seconds\n")
            f.write("===============================\n\n")

        if not results:
            return jsonify({"error": "No valid content found in results"}), 404

        # Stage 5: Summarization
        time_before = time.time()
        summary = summarizer(initial_query, results)
        
        # Record timing
        time_after = time.time()
        time_taken = time_after - time_before
        with open('timestamp.txt', 'a', encoding='utf-8') as f:
            f.write("========= SUMMARIZATION =========\n")
            f.write(f"Time taken for summarization: {time_taken:.4f} seconds\n")
            f.write("=================================\n\n")

        # Overall Time Tracking
        overall_end = time.time()
        overall_time_taken = overall_end - overall_start
        with open('timestamp.txt', 'a', encoding='utf-8') as f:
            f.write("========= OVERALL OPERATION =========\n")
            f.write(f"Total time taken: {overall_time_taken:.4f} seconds\n")
            f.write("=====================================\n\n")
        
        return jsonify(summary)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/socket.io/', defaults={'path': ''})
@app.route('/socket.io/<path:path>')
def catch_socket_io(path):
    return '', 204


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5001, debug=True)
