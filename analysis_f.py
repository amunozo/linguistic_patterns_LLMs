import numpy as np
import pandas as pd
from datetime import datetime



def limit_length(sentences, lower, upper):
    return [sentence for sentence in sentences if lower <= len(sentence) <= upper]

def myconverter(o):
    if isinstance(o, datetime):
        return o.__str__()
    
def create_length_dict(sentences_dict, max_length, step):
    length_dict = {}
    for i in range(1, max_length, step):
        key = f'{i}_{i+step-1}' if i+step-1 < max_length else f'{i}_'
        length_dict[key] = {model: limit_length(sentences_dict[model], i, i+step-1) for model in sentences_dict.keys()}
    return length_dict

def stats_df(dict_of_sentences):
    df = pd.DataFrame(columns=['lm', '% l', '% r', 'avg_len', 'avg_r_len', 'avg_l_len', 'std_len', 'std_r_len', 'std_l_len', 'n_sentences'])
    
    for lm in dict_of_sentences.keys():
        list_of_sentences = dict_of_sentences[lm]
        left = 0
        right = 0
        abs_length = []
        right_lenght = []
        left_length = []
        n_sentences = 0


        for sentence in list_of_sentences:
            n_sentences += 1
            for line in sentence:
                if line['head'] != '0':
                    if line['id'] < line['head']:
                        left += 1
                        left_length.append(line['head'] - line['id'])
                    else:
                        right += 1
                        right_lenght.append(line['id'] - line['head'])
                    
                    abs_length.append(abs(line['head'] - line['id']))

        avg_l = left/len(abs_length) * 100
        avg_r = right/len(abs_length) * 100
        avg_len = sum(abs_length)/len(abs_length)
        avg_r_len = sum(right_lenght)/len(right_lenght)
        avg_l_len = sum(left_length)/len(left_length)
        std_len = np.std(abs_length)
        std_r_len = np.std(right_lenght)             
        std_l_len = np.std(left_length)

        new_row =  {
                'lm': lm,
                '% l': avg_l,
                '% r': avg_r,
                'avg_len': avg_len,
                'avg_r_len': avg_r_len,
                'avg_l_len': avg_l_len,
                'std_len': std_len,
                'std_r_len': std_r_len,
                'std_l_len': std_l_len,
                'n_sentences': n_sentences
            }
        
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    return df