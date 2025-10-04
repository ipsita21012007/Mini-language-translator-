from transformers import MarianMTModel, MarianTokenizer
import torch

class MiniTranslator:
    def __init__(self, model_path=None, model_name=None):
        if model_path:
            self.model = MarianMTModel.from_pretrained(model_path)
            self.tokenizer = MarianTokenizer.from_pretrained(model_path)
        elif model_name:
            self.model = MarianMTModel.from_pretrained(model_name)
            self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        else:
            raise ValueError("Either model_path or model_name must be provided")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
    
    def translate(self, text, max_length=128):
        # Tokenize input text
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=max_length
        ).to(self.device)
        
        # Generate translation
        with torch.no_grad():
            translated_tokens = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=4,
                early_stopping=True
            )
        
        # Decode the translated text
        translated_text = self.tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
        
        return translated_text

# Pre-loaded translators for common language pairs
class TranslationService:
    def __init__(self):
        self.models = {}
        self.available_models = {
            'en-hi': 'Helsinki-NLP/opus-mt-en-hi',
            'hi-en': 'Helsinki-NLP/opus-mt-hi-en',
            'en-ta': 'Helsinki-NLP/opus-mt-en-ta',
            'ta-en': 'Helsinki-NLP/opus-mt-ta-en',
            'en-te': 'Helsinki-NLP/opus-mt-en-te',
            'te-en': 'Helsinki-NLP/opus-mt-te-en',
            'en-mr': 'Helsinki-NLP/opus-mt-en-mr',
            'mr-en': 'Helsinki-NLP/opus-mt-mr-en'
        }
    
    def load_model(self, language_pair):
        if language_pair not in self.available_models:
            raise ValueError(f"Unsupported language pair: {language_pair}")
        
        if language_pair not in self.models:
            model_name = self.available_models[language_pair]
            self.models[language_pair] = MiniTranslator(model_name=model_name)
        
        return self.models[language_pair]
    
    def translate(self, text, source_lang, target_lang):
        language_pair = f"{source_lang}-{target_lang}"
        translator = self.load_model(language_pair)
        return translator.translate(text)