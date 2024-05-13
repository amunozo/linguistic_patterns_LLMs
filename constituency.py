from analysis_f import myconverter
import json
import stanza
from tqdm import tqdm
import random
from transformers import pipeline

from argparse import ArgumentParser
from nltk.tree import Tree
from collections import Counter
import codecs
import copy
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
import os

def get_tree_spans(tree, root, ignore_non_terminal=False):
    
    spans = []
    if type(tree) != type(""):
        if not root:  
               
            if len(tree.leaves()) == 1:
                if "@" in tree.label():
                    spans = [(tree.leaves(), tree.label())]
            else:  
                if ignore_non_terminal:
                    symbol = "-"
                else:
                    symbol = tree.label()
                spans = [(tree.leaves(), symbol)]
            
        for child in tree:            
            if type(child) != type(""):
                spans.extend(get_tree_spans(child, False))

    return spans

def avg_non_terminal_len(path_file):
    
    with codecs.open(path_file) as f:    
        gold_trees = f.readlines()
        non_terminals = []
        nt_lengths = {}
    
    for gold_tree in gold_trees:
        gold_tree =Tree.fromstring(gold_tree,remove_empty_top_bracketing=True)
        gold_tree.collapse_unary(collapsePOS=True, collapseRoot=True, joinChar="@")  
        gold_spans = get_tree_spans(gold_tree,True)
        
        for span_text, span_nt in gold_spans:
            
            uppermost_span_nt = span_nt.split("@")[0]
            non_terminals.append(uppermost_span_nt)
            if uppermost_span_nt not in nt_lengths:
                nt_lengths[uppermost_span_nt] = []
            
            nt_lengths[uppermost_span_nt].append(len(span_text))

    for key in nt_lengths:
        nt_lengths[key] = float(sum(nt_lengths[key]) / len(nt_lengths[key]))
    nt_counter = Counter(non_terminals)

    return nt_counter, nt_lengths


if __name__ == "__main__":
    nlp = stanza.Pipeline('en', use_gpu=True, device_map="auto")
    #classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", device_map="auto")


    # Batch processing setup
    batch_size = 1  # Adjust the batch size based on your GPU memory and requirements

    data_dict = {
        'original': json.load(open('postmistral_data/original/articles_parsed.json', 'r')),
        'mistral_7B': json.load(open('postmistral_data/mistral_7B/articles_complete.json', 'r')),
        'falcon_7B': json.load(open('postmistral_data/falcon_7B/articles_complete.json', 'r')),
        'llama_7B': json.load(open('postmistral_data/llama_7B/articles_complete.json', 'r')),
        'llama_13B': json.load(open('postmistral_data/llama_13B/articles_complete.json', 'r')),
        'llama_30B': json.load(open('postmistral_data/llama_30B/articles_complete.json', 'r')),
        'llama_65B': json.load(open('postmistral_data/llama_65B/articles_complete.json', 'r')),
    }

    for lm in data_dict.keys():
        print(f'Processing {lm}...')

        # Process in batches
        for i in tqdm(range(0, len(data_dict[lm]), batch_size)):
            batch_articles = data_dict[lm][i:i + batch_size]
            batch_texts = [article['lead_paragraph'] for article in batch_articles]

            # Process each batch
            docs = [nlp(text) for text in batch_texts]
            #emotions = classifier(batch_texts)

            for j, article in enumerate(batch_articles):
                article['constituents'] = [str(sentence.constituency) for sentence in docs[j].sentences]
                #article['parsed'] = docs[j].to_dict()  # type: ignore
                #article['emotion'] = emotions[j]['label']

        with open(f'postmistral_data/{lm}/articles.json', 'w') as f:
            json.dump(data_dict[lm], f, default = myconverter)