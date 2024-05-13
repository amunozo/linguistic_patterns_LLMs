import torch
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import stanza


class Analyzer():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def __init__(self, device: str):
        self.device  = device
        self.kwargs = None
        self.parser = stanza.Pipeline(lang='en', use_gpu=True, device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained('j-hartmann/emotion-english-distilroberta-base')
        self.classifier = AutoModelForSequenceClassification.from_pretrained('j-hartmann/emotion-english-distilroberta-base').to(self.device)


    def read_articles(self, filepath):
        with open(filepath, 'r') as file:
            self.articles = json.load(file)[:10]

    def parse(self, articles):
        out_docs = self.parser.bulk_process([article['lead_paragraph'] for article in articles])
        parsed = [doc.to_dict() for doc in out_docs]
        constituents = [str(sentence.constituency) for doc in out_docs for sentence in doc.sentences]
        self.parsed, self.constituents = parsed, constituents
        self.constituents = constituents

    def _classify(self, article: dict):
        inputs = self.tokenizer(article['lead_paragraph'], return_tensors="pt", padding=True, truncation=True).to(self.device)
        outputs = self.classifier(**inputs)
        label = torch.argmax(outputs.logits, dim=-1).item()
        emotion = self.classifier.config.id2label[label]

        return emotion
    
    def batch_classify(self, batch_size: int):
        with tqdm(total=len(self.articles), desc='Articles') as bar:
            dict = {'parsed': [], 'emotion': [], 'constituents': []}
            for i in range(0, len(self.articles), batch_size):
                batch_articles = self.articles[i:i + batch_size]
                #batch_parsed = self.parsed[i:i + batch_size]
                batch_constituents = self.constituents[i:i + batch_size]
                #batch_emotion = [self._classify(article) for article in batch_articles]
                #dict['parsed'].append(batch_parsed)
                #dict['emotion'].append(batch_emotion)
                dict['constituents'].append(batch_constituents)
                print(batch_constituents)
                bar.update(len(batch_articles))
            
        # Flatten the lists in dict
        #flat_parsed = [item for sublist in dict['parsed'] for item in sublist]
        #flat_emotion = [item for sublist in dict['emotion'] for item in sublist]
        flat_constituents = [item for sublist in dict['constituents'] for item in sublist]

        # add to articles
        #self.articles = [{**article, 'emotion': emotion, 'parsed': parsed, 'constituents': constituents} 
        #                for article, emotion, parsed, constituents in 
        #                zip(self.articles, flat_emotion, flat_parsed, flat_constituents)]
        self.articles = [{**article, 'constituents': constituents} 
                        for article, constituents in 
                        zip(self.articles, self.constituents)]
                

                

def main():
    lms = ['original'] #, 'mistral_7B', 'falcon_7B', 'llama_7B', 'llama_13B', 'llama_30B', 'llama_65B']
    analyzer = Analyzer(device='cuda')
    for lm in tqdm(lms):
        print(f'Processing {lm}...')
        if lm == 'original':
            analyzer.read_articles(f'postmistral_data/{lm}/articles_parsed.json')
        else:
            analyzer.read_articles(f'postmistral_data/{lm}/articles_complete.json')

        analyzer.parse(analyzer.articles)
        analyzer.batch_classify(batch_size=2048)

        with open(f'postmistral_data/{lm}/TEST.json', 'w') as file:
            json.dump(analyzer.articles, file)
        print(f'Processed {lm}.')


if __name__ == '__main__':
    main()