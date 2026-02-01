#!/usr/bin/env python3
"""Test script to verify OpenRouter client works with different models."""

import os
from openrouter_client import OpenAICompatibleChatClient

def test_model(model_name: str, description: str):
    """Test a specific model configuration."""
    print(f"\n{'='*60}")
    print(f"Testing: {description}")
    print(f"Model: {model_name}")
    print(f"{'='*60}")
    
    try:
        client = OpenAICompatibleChatClient(model=model_name)
        
        messages = [
            {"role": "user", "content": "What is 2+2? Answer in one word only."}
        ]
        
        result = client.complete(messages, temperature=0.0)
        print(f"✓ SUCCESS")
        print(f"Response: {result}")
        return True
        
    except Exception as e:
        print(f"✗ FAILED")
        print(f"Error: {e}")
        return False

def main():
    # Check if API key is set
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("ERROR: OPENROUTER_API_KEY environment variable not set")
        print("Please run: export OPENROUTER_API_KEY='your-key-here'")
        return
    
    print("\n" + "="*60)
    print("OpenRouter Client Test Suite")
    print("="*60)
    
    # Test different model configurations
    tests = [
        ("deepseek/deepseek-r1", "DeepSeek R1 (should NOT force provider)"),
    #   ("deepseek-ai/DeepSeek-Prover-V2-671B:novita", "DeepSeek Prover with explicit provider"),
        ("openai/gpt-3.5-turbo", "OpenAI GPT-3.5 (should NOT force provider)"),
        ("anthropic/claude-3-haiku", "Claude Haiku (should NOT force provider)"),
    ]
    
    results = []
    for model, desc in tests:
        success = test_model(model, desc)
        results.append((desc, success))
    
    # Summary
    print(f"\n{'='*60}")
    print("Test Summary")
    print(f"{'='*60}")
    for desc, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {desc}")
    
    total = len(results)
    passed = sum(1 for _, s in results if s)
    print(f"\nTotal: {passed}/{total} tests passed")

if __name__ == "__main__":
    main()
