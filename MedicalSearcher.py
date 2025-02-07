from typing import List, Dict, Tuple
import numpy as np
from collections import defaultdict
import math
import faiss
from scispacy.linking import EntityLinker
import spacy
from transformers import AutoTokenizer, AutoModel
import torch
from itertools import chain
import requests

class MedicalSearcher:
    def __init__(self):
        # Load biomedical NLP models
        self.nlp = spacy.load("en_core_sci_scibert")  # Scientific/biomedical model
        self.linker = EntityLinker(resolve_abbreviations=True)
        self.nlp.add_pipe("scispacy_linker")
        
        # Load PubMedBERT for biomedical text
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
        self.bert_model = AutoModel.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
        
        # Initialize FAISS indices
        self.semantic_index = None
        self.tfidf_index = None
        self.doc_map = []
        
        # Medical concept mappings
        self.umls_to_doc = defaultdict(list)
        self.doc_to_umls = defaultdict(set)

    def _extract_medical_concepts(self, text: str) -> List[str]:
        """Extract UMLS concepts and medical entities from text"""
        request = {"text": text}
        response = requests.post("https://a2fe4k5ex3iwhv-5000.proxy.runpod.net/match", json=request)
        entities = response.json()
        
        # Extract UMLS concepts
        concepts = []
        for ent in entities:
            if ent["similarity"] > 0.5:
                # Get UMLS CUI (Concept Unique Identifier)
                [0]
                concepts.append(cui)
        
        return concepts

    def _create_bert_embedding(self, text: str) -> np.ndarray:
        """Create biomedical BERT embedding for text"""
        inputs = self.tokenizer(text, return_tensors="pt", 
                              max_length=512, truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
            # Use [CLS] token embedding as document representation
            embedding = outputs.last_hidden_state[:, 0, :].numpy()
        return embedding

    def _calculate_medical_tf(self, text: str) -> Dict[str, float]:
        """Calculate term frequencies with medical term weighting"""
        doc = self.nlp(text)
        term_weights = defaultdict(float)
        
        for token in doc:
            weight = 1.0
            # Increase weight for medical terms
            if token._.is_entity:
                weight = 2.0
            # Further increase for drug names
            if token._.is_drug:
                weight = 3.0
            term_weights[token.text.lower()] += weight
            
        return term_weights

    def fit(self, documents: List[str]):
        """Build search indices from documents"""
        num_docs = len(documents)
        
        # Extract features and build indices
        bert_embeddings = []
        tfidf_vectors = []
        
        for i, doc in enumerate(documents):
            # Store document mapping
            self.doc_map.append(doc)
            
            # Extract and store UMLS concepts
            concepts = self._extract_medical_concepts(doc)
            for concept in concepts:
                self.umls_to_doc[concept].append(i)
                self.doc_to_umls[i].add(concept)
            
            # Create BERT embedding
            bert_emb = self._create_bert_embedding(doc)
            bert_embeddings.append(bert_emb)
            
            # Create TF-IDF vector
            tfidf = self._calculate_medical_tf(doc)
            tfidf_vectors.append(tfidf)

        # Build FAISS indices
        dim = bert_embeddings[0].shape[1]
        self.semantic_index = faiss.IndexFlatIP(dim)
        self.semantic_index.add(np.vstack(bert_embeddings))

        return self

    def search(self, query: str, k: int = 5) -> List[Dict]:
        """
        Search for relevant documents using hybrid retrieval
        """
        # Extract query features
        query_concepts = self._extract_medical_concepts(query)
        query_bert = self._create_bert_embedding(query)
        
        # Semantic search
        semantic_sims, semantic_ids = self.semantic_index.search(query_bert, k*2)
        
        # Concept-based filtering
        concept_boosted_results = []
        for sim, doc_id in zip(semantic_sims[0], semantic_ids[0]):
            boost = 1.0
            # Boost score if document shares medical concepts with query
            shared_concepts = len(self.doc_to_umls[doc_id].intersection(query_concepts))
            if shared_concepts > 0:
                boost = 1.0 + (0.1 * shared_concepts)
            
            concept_boosted_results.append({
                'doc_id': doc_id,
                'score': sim * boost,
                'content': self.doc_map[doc_id]
            })
        
        # Sort and return top k results
        results = sorted(concept_boosted_results, key=lambda x: x['score'], reverse=True)[:k]
        return results

def rank_results_with_medical_search(model, user_query: str, all_query_results: List[Dict], 
                                   k: int = 5) -> List[Dict]:
    """
    Rank search results using medical-specific search
    """
    # Initialize searcher
    searcher = MedicalSearcher()
    
    # Prepare documents
    documents = []
    content_to_result_map = {}
    
    for query_result in all_query_results:
        for result in query_result.get('query_search_results', []):
            content = result.get('content', '').strip()
            if content:
                documents.append(content)
                content_to_result_map[content] = result
    
    # Fit searcher and get results
    searcher.fit(documents)
    search_results = searcher.search(user_query, k=k)
    
    # Format results
    formatted_results = []
    for res in search_results:
        content = res['content']
        result = content_to_result_map[content]
        formatted_results.append({
            'similarity': res['score'],
            'result': result
        })
    
    return formatted_results