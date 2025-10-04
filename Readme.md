# Mini Language Translator (ML)

A web-based translator for English ↔ Regional Indian Languages (Hindi, Tamil, Telugu, Marathi) using Hugging Face pretrained models.

## Features

- Translate between English and multiple Indian languages
- Clean, responsive web interface
- Uses state-of-the-art transformer models
- Real-time translation
- Language swapping functionality

## Supported Languages

- English (en)
- Hindi (hi)
- Tamil (ta)
- Telugu (te)
- Marathi (mr)

## Installation

1. Extract the folder to your desktop
2. Open terminal/command prompt in the folder
3. Install requirements:
```bash
pip install -r requirements.txt

## 🔀 Translation Methods

You can choose between two translation engines:

1. **Seq2Seq Model**  
   - Custom-trained LSTM encoder-decoder  
   - Requires training on sentence pairs  
   - Good for small, focused datasets

2. **Hugging Face Pretrained Model**  
   - Uses models like `Helsinki-NLP/opus-mt-en-hi`  
   - No training required  
   - Supports multiple Indian languages

Select your method in the web app or via command-line flag.