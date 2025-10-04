# run_app.py
from transformers import MarianMTModel, MarianTokenizer
from flask import Flask, render_template, request, jsonify
import torch
import os

app = Flask(__name__)

class TranslationService:
    def __init__(self):
        self.models = {}
        self.available_models = {
            'en-hi': 'Helsinki-NLP/opus-mt-en-hi',
            'hi-en': 'Helsinki-NLP/opus-mt-hi-en',
            'en-ta': 'Helsinki-NLP/opus-mt-en-ta',
            'ta-en': 'Helsinki-NLP/opus-mt-ta-en',
        }
    
    def load_model(self, language_pair):
        if language_pair not in self.available_models:
            raise ValueError(f"Unsupported language pair: {language_pair}")
        
        if language_pair not in self.models:
            print(f"Loading model for {language_pair}...")
            model_name = self.available_models[language_pair]
            self.models[language_pair] = {
                'tokenizer': MarianTokenizer.from_pretrained(model_name),
                'model': MarianMTModel.from_pretrained(model_name)
            }
            print(f"Model {language_pair} loaded successfully!")
        
        return self.models[language_pair]
    
    def translate(self, text, source_lang, target_lang):
        language_pair = f"{source_lang}-{target_lang}"
        model_data = self.load_model(language_pair)
        
        inputs = model_data['tokenizer'](text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        
        with torch.no_grad():
            translated_tokens = model_data['model'].generate(
                **inputs,
                max_length=128,
                num_beams=4,
                early_stopping=True
            )
        
        translated_text = model_data['tokenizer'].decode(translated_tokens[0], skip_special_tokens=True)
        return translated_text

translator_service = TranslationService()

# Supported languages
LANGUAGES = {
    'en': 'English',
    'hi': 'Hindi', 
    'ta': 'Tamil'
}

@app.route('/')
def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Mini Language Translator</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .container { max-width: 800px; margin: 0 auto; }
            .input-area { margin: 20px 0; }
            textarea { width: 100%; height: 100px; margin: 10px 0; }
            button { padding: 10px 20px; background: #007bff; color: white; border: none; cursor: pointer; }
            .result { margin: 20px 0; padding: 15px; background: #f8f9fa; border-radius: 5px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🌍 Mini Language Translator</h1>
            
            <div class="input-area">
                <label>From:</label>
                <select id="source-lang">
                    <option value="en">English</option>
                    <option value="hi">Hindi</option>
                    <option value="ta">Tamil</option>
                </select>
                
                <label>To:</label>
                <select id="target-lang">
                    <option value="hi">Hindi</option>
                    <option value="en">English</option>
                    <option value="ta">Tamil</option>
                </select>
            </div>
            
            <textarea id="source-text" placeholder="Enter text to translate..."></textarea>
            <button onclick="translateText()">Translate</button>
            <textarea id="translated-text" placeholder="Translation will appear here..." readonly></textarea>
            
            <div id="result"></div>
        </div>

        <script>
            async function translateText() {
                const sourceText = document.getElementById('source-text').value;
                const sourceLang = document.getElementById('source-lang').value;
                const targetLang = document.getElementById('target-lang').value;
                const resultDiv = document.getElementById('result');
                
                if (!sourceText.trim()) {
                    resultDiv.innerHTML = '<div style="color: red;">Please enter some text to translate.</div>';
                    return;
                }
                
                try {
                    const response = await fetch('/translate', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            text: sourceText,
                            source_lang: sourceLang,
                            target_lang: targetLang
                        })
                    });
                    
                    const data = await response.json();
                    
                    if (response.ok) {
                        document.getElementById('translated-text').value = data.translated_text;
                        resultDiv.innerHTML = `<div style="color: green;">Translation successful!</div>`;
                    } else {
                        resultDiv.innerHTML = `<div style="color: red;">Error: ${data.error}</div>`;
                    }
                } catch (error) {
                    resultDiv.innerHTML = `<div style="color: red;">Network error: ${error}</div>`;
                }
            }
        </script>
    </body>
    </html>
    """

@app.route('/translate', methods=['POST'])
def translate():
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        source_lang = data.get('source_lang', 'en')
        target_lang = data.get('target_lang', 'hi')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        print(f"Translating: '{text}' from {source_lang} to {target_lang}")
        translated_text = translator_service.translate(text, source_lang, target_lang)
        print(f"Translation result: '{translated_text}'")
        
        return jsonify({
            'original_text': text,
            'translated_text': translated_text,
            'source_lang': source_lang,
            'target_lang': target_lang
        })
    
    except Exception as e:
        print(f"Translation error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting Mini Language Translator...")
    print("Available on: http://localhost:5000")
    print("Supported languages: English, Hindi, Tamil")
    app.run(debug=True, host='0.0.0.0', port=5000)