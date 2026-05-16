"""
Offline Evaluation Framework — Recall@K and Groundedness Testing.

This script runs test queries against the Hybrid Retrieval system to calculate:
1. Recall@K (Is the expected assessment in the top K results?)
2. Groundedness (Are all recommended assessments actually from the catalog?)
"""

import json
import os
import sys

# Ensure absolute path is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.services.chat_service import ChatService
from app.services.catalog_service import catalog_service

def load_golden_dataset(filepath: str) -> list:
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def run_evaluation():
    # 1. Initialize services
    print("Initializing services...")
    catalog_service.load()
    chat_service = ChatService()
    
    # Wait for async retrieval service to load chroma
    print("Loading vector database...")
    
    dataset = load_golden_dataset(os.path.join(os.path.dirname(__file__), "golden_queries.json"))
    
    total_queries = len(dataset)
    hits_at_3 = 0
    hits_at_10 = 0
    total_recommendations = 0
    grounded_recommendations = 0

    print(f"\nRunning evaluation on {total_queries} golden queries...\n")

    for i, test_case in enumerate(dataset, 1):
        query = test_case["query"]
        expected_url = test_case["expected_url"]
        
        # Simulate an incoming chat request
        messages = [{"role": "user", "content": query}]
        
        # We call the internal _search directly to bypass clarification thresholds
        state = chat_service._reconstruct_state(messages)
        recommendations = chat_service._search(state, max_results=10)
        
        urls = [r.url for r in recommendations]
        
        total_recommendations += len(urls)
        # Groundedness Check (against actual catalog)
        grounded = [u for u in urls if catalog_service.get_by_url(u) is not None]
        grounded_recommendations += len(grounded)
        
        # Recall Check
        is_hit_3 = expected_url in urls[:3]
        is_hit_10 = expected_url in urls[:10]
        
        if is_hit_3:
            hits_at_3 += 1
        if is_hit_10:
            hits_at_10 += 1
            
        print(f"[{i}/{total_queries}] Query: '{query}'")
        print(f"  Expected: {expected_url}")
        print(f"  Recall@3: {'PASS' if is_hit_3 else 'FAIL'} | Recall@10: {'PASS' if is_hit_10 else 'FAIL'}\n")

    recall_3 = (hits_at_3 / total_queries) * 100
    recall_10 = (hits_at_10 / total_queries) * 100
    groundedness = (grounded_recommendations / total_recommendations * 100) if total_recommendations > 0 else 100

    print("="*40)
    print("      EVALUATION RESULTS")
    print("="*40)
    print(f"Total Queries Tested: {total_queries}")
    print(f"Recall@3:  {recall_3:.1f}%")
    print(f"Recall@10: {recall_10:.1f}%")
    print(f"Groundedness: {groundedness:.1f}% (Zero Hallucination Guard Active)")
    print("="*40)

if __name__ == "__main__":
    run_evaluation()
