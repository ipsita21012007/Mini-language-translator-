# app/main.py
from flask import Flask, render_template, request, jsonify
import sys
import os

# Add the models directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.translator import TranslationService

app = Flask(__name__)
translator_service = TranslationService()

# Supported languages
LANGUAGES = {
    'en': 'English',
    'hi': 'Hindi',
    'ta': 'Tamil',
    'te': 'Telugu',
    'mr': 'Marathi'
}

@app.route('/')
def index():
    return render_template('index.html', languages=LANGUAGES)

@app.route('/translate', methods=['POST'])
def translate_text():
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        source_lang = data.get('source_lang', 'en')
        target_lang = data.get('target_lang', 'hi')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Translate the text
        translated_text = translator_service.translate(text, source_lang, target_lang)
        
        return jsonify({
            'original_text': text,
            'translated_text': translated_text,
            'source_lang': source_lang,
            'target_lang': target_lang
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/languages')
def get_languages():
    return jsonify(LANGUAGES)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)