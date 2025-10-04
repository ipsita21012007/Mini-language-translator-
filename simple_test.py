# simple_test.py
from transformers import MarianMTModel, MarianTokenizer
import torch

def test_translation():
    print("Testing translation engine...")
    
    # Test English to Hindi
    print("Loading English to Hindi model...")
    tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-hi")
    model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-hi")
    
    text = "Hello, how are you?"
    print(f"Translating: '{text}'")
    
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    
    with torch.no_grad():
        translated_tokens = model.generate(**inputs)
    
    result = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
    print(f"English: {text}")
    print(f"Hindi: {result}")
    print("✅ Translation test successful!")

if __name__ == "__main__":
    test_translation()