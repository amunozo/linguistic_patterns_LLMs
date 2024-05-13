from transformers import AutoModelForCausalLM, AutoTokenizer
import datetime
import json
from tqdm import tqdm
import os
import warnings

# Ignore all user warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Set environment variable for model cache directory
os.environ["HF_HOME"] = "path_to_huggingface_cache"

# Replace with your actual API key
apikey = "YOUR_API_KEY_HERE"

# Language models for processing
lms = [
    'hf/7B',
    'hf/13B',
    'hf/30B',
    'hf/65B',
    "mistralai/Mistral-7B-v0.1",
    "tiiuae/falcon-7b",
]

# Load article data
articles = json.load(open('path_to_def_articles/original.json'))

# Process articles with each language model
for lm in lms:
    try:
        model = AutoModelForCausalLM.from_pretrained(lm, device_map="auto", load_in_8bit=True)
        tokenizer = AutoTokenizer.from_pretrained(lm)

        # Handle lm name for output file
        lm_name = lm.split('/')[-1] if '/' in lm else lm
        filename = f"output/{lm_name}.json"

        print('-' * 100)
        print(f'MODEL: {lm_name}')
        print('-' * 100)

        output_data = []
        for index, article in enumerate(tqdm(articles), start=1):
            article_data = {
                'id': index,
                'headline': article['headline'],
                'lead_paragraph': article['lead_paragraph'],
                'pub_date': article['pub_date'],
                'web_url': article['web_url']
            }

            prompt = f'"{article["abstract"]}"\n{" ".join(article["lead_paragraph"].split(" ")[:3])} '
            input_ids = tokenizer.encode(prompt, return_tensors='pt').to("cuda")

            sample_outputs = model.generate(
                input_ids,
                max_length=200,
                do_sample=True,
                top_p=0.9, 
                early_stopping=True,
                temperature=0.7,
                num_return_sequences=1,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id
            )

            generations = []
            for i, sample_output in enumerate(sample_outputs, start=1):
                output = tokenizer.decode(sample_output, skip_special_tokens=True)
                generations.append({
                    'id': i,
                    'headline': article['headline']['main'],
                    'output': output.split('\n')[1]
                })

            article_data['generations'] = generations
            output_data.append(article_data)

        with open(filename, "w") as f:
            json.dump(output_data, f)

    except Exception as e:
        print(f'ERROR: {lm_name} - {str(e)}')
        continue
