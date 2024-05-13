import torch
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import stanza


class Analyzer:
    def __init__(self, device: str = None):
        self.device = device if device else 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer = AutoTokenizer.from_pretrained('j-hartmann/emotion-english-distilroberta-base')
        self.classifier = AutoModelForSequenceClassification.from_pretrained('j-hartmann/emotion-english-distilroberta-base').to(self.device)
        self.parser = stanza.Pipeline(lang='en', use_gpu=True, device_map="auto")
        self.articles = []
        self.parsed = []
        self.constituents = []

    def read_articles(self, filepath):
        with open(filepath, 'r') as file:
            self.articles = json.load(file)[:10]  # Load only first 10 for testing or example

    def parse(self):
        articles_text = [article['lead_paragraph'] for article in self.articles]
        out_docs = self.parser.bulk_process(articles_text)
        self.parsed = [doc.to_dict() for doc in out_docs]
        self.constituents = [str(sentence.constituency) for doc in out_docs for sentence in doc.sentences]

    def classify_emotions(self):
        for article in self.articles:
            inputs = self.tokenizer(article['lead_paragraph'], return_tensors="pt", padding=True, truncation=True).to(self.device)
            outputs = self.classifier(**inputs)
            label = torch.argmax(outputs.logits, dim=-1).item()
            article['emotion'] = self.classifier.config.id2label[label]

    def batch_classify(self, batch_size: int):
        # Batch process articles for efficiency
        with tqdm(total=len(self.articles), desc='Classifying Articles') as progress_bar:
            for start_index in range(0, len(self.articles), batch_size):
                self.classify_emotions()
                progress_bar.update(batch_size)

    def process_articles(self, filepath, output_path):
        self.read_articles(filepath)
        self.parse()
        self.batch_classify(batch_size=2048)
        with open(output_path, 'w') as file:
            json.dump(self.articles, file)

def main():
    analyzer = Analyzer(device='cuda')
    models = ['original']  # Extend this list with other models if needed
    for model in tqdm(models, desc="Processing Models"):
        input_file = f'postmistral_data/{model}/articles_parsed.json'
        output_file = f'postmistral_data/{model}/TEST.json'
        print(f'Processing {model}...')
        analyzer.process_articles(input_file, output_file)
        print(f'Processed {model}.')

if __name__ == '__main__':
    main()