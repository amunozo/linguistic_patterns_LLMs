import json
import torch, re, os
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datetime import datetime
from typing import List

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"

PAD = '<pad>'

def flatten(list_of_lists):
    result = list()
    for item in list_of_lists:
        result += item 
    return result

def load_data(filepath):
    with open(filepath, 'r') as file:
        return json.load(file)

def save_data(data, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as file:
        json.dump(data, file, default=json_converter)

def json_converter(o):
    if isinstance(o, datetime):
        return o.__str__()


class Generator:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def __init__(self, model_name: str, device: torch.device, articles: str):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left', truncation_size='left')
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        try:
            self.generator = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', load_in_8bit=True)
        except OSError:
            self.generator = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', load_in_8bit=False)
            print('No 8bit model available, loading full precision model instead')
        self.model_size = self.model_name.split('/')[-1] if 'llama' in self.model_name else '7B'
        self.device = device
        self.articles = self.read_articles(articles)
        self.kwargs = None

    def read_articles(self, filepath):
        with open(filepath, 'r') as file:
            return json.load(file)
    
    def get_contexts(self, article: dict) -> List[str]:
            context = f'''"{article['headline']['main']}"\n{' '.join(article['lead_paragraph'].split(' ')[:3])}'''
            return context
    
    def _generate(self, context: str) -> str:
        # get input ids
        tokens = self.tokenizer(context, padding=True, return_tensors='pt', add_special_tokens=False, return_attention_mask=True)
        inputs, mask = tokens['input_ids'].cuda(), tokens['attention_mask'].cuda()

        # get outputs
        out = self.generator.generate(
            inputs, attention_mask=mask,
            max_length=200,
            do_sample=True,
            top_p=0.9, temperature=0.7,
            #early_stopping=True,
            num_return_sequences=1, repetition_penalty=1.1,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        out = [x.split('\n')[1] for x in self.tokenizer.batch_decode(out.tolist(), skip_special_tokens=True)]
        return out
    
    def estimate_batch_size(self):
        batch_size = {
            '7B': 512,   # Example values: 2 GB for batch size of 32
            '13B': 256,
            '30B': 128,
            '65B': 32
        }

        return batch_size[self.model_size]


    def batch_generate(self, batch_size: int): #, batch_size: int): #, save: str):
        print(f'Total number of sentences: {len(self.articles)}')
        # get contexts
        contexts = [self.get_contexts(article) for article in self.articles]
        with tqdm(total=len(self.articles), desc='Articles') as bar:
            specs = []
            for i in range(0, len(contexts), batch_size):
                specs += self._generate(contexts[i:i+batch_size])
                bar.update(len(specs))
        return specs

def main():
    lms = {
        #'mistral_7B': 'mistralai/Mistral-7B-v0.1',
        #'falcon_7B': 'tiiuae/falcon-7b',
        #'llama_7B': '/home/lys/llama_weights/hf/7B',
        #'llama_13B': '/home/lys/llama_weights/hf/13B',
        #'llama_30B': '/home/lys/llama_weights/hf/30B',
        'llama_65B': '/home/lys/llama_weights/hf/65B',
    }
    # read file at postmistral_data/original/articles.json
    articles = 'postmistral_data/original/articles.json'
    for model in lms:
        print('Generating for model {}'.format(model))
        if not os.path.exists('postmistral_data/' + model):
            os.mkdir('postmistral_data/' + model)
        generator = Generator(model_name=lms[model], device='cuda', articles = articles)
        batch_size = generator.estimate_batch_size()
        try:
            sentences = generator.batch_generate(batch_size)
        except torch.cuda.memory.OutOfMemoryError:
            print('Out of memory, trying again with smaller batch size')
            batch_size = batch_size // 2
            sentences = generator.batch_generate(batch_size)

        print('Finished generating sentences')
        assert len(sentences) == len(generator.articles)

        for i in range(len(sentences)):
            generator.articles[i]['lead_paragraph'] = sentences[i]
        
        with open('postmistral_data/' + model + '/articles.json', 'w') as f:
            json.dump(generator.articles, f)
        
        # clear GPU memory
        del generator
        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()