# train.py
import pandas as pd
import torch
from transformers import MarianMTModel, MarianTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import Dataset
import os

class TranslationDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, data, source_lang, target_lang, max_length=128):
        self.tokenizer = tokenizer
        self.data = data
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        source_text = str(self.data.iloc[idx][self.source_lang])
        target_text = str(self.data.iloc[idx][self.target_lang])
        
        source_encodings = self.tokenizer(
            source_text, 
            max_length=self.max_length, 
            padding='max_length', 
            truncation=True,
            return_tensors="pt"
        )
        
        target_encodings = self.tokenizer(
            target_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        
        labels = target_encodings['input_ids']
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': source_encodings['input_ids'].flatten(),
            'attention_mask': source_encodings['attention_mask'].flatten(),
            'labels': labels.flatten()
        }

def train_translator(model_name, csv_file, source_lang, target_lang, output_dir):
    # Load data
    data = pd.read_csv(csv_file)
    print(f"Loaded {len(data)} sentence pairs")
    
    # Load pretrained model and tokenizer
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    
    # Create dataset
    dataset = TranslationDataset(tokenizer, data, source_lang, target_lang)
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=10,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        predict_with_generate=True,
        fp16=False,
        load_best_model_at_end=True,
    )
    
    # Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Save model
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")

if __name__ == "__main__":
    # For English to Hindi
    print("Training English to Hindi model...")
    train_translator(
        model_name="Helsinki-NLP/opus-mt-en-hi",
        csv_file="data/english_hindi_pairs.csv",
        source_lang="english",
        target_lang="hindi",
        output_dir="models/en_hi_model"
    )
    
    # For English to Tamil
    print("Training English to Tamil model...")
    train_translator(
        model_name="Helsinki-NLP/opus-mt-en-ta",
        csv_file="data/english_tamil_pairs.csv",
        source_lang="english",
        target_lang="tamil",
        output_dir="models/en_ta_model"
    )