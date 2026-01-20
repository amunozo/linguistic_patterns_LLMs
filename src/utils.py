import pandas as pd

def stats_df(treebanks):
    # create df with columns: treebank, avg_l, avg_r, avg_len, avg_r_len, avg_l_len
    df = pd.DataFrame(columns=['treebank', '% l', '% r', 'avg_len', 'avg_r_len', 'avg_l_len', 'std_len', 'std_r_len'])
    for tb in treebanks:
        left = 0
        right = 0
        abs_length = []
        right_lenght = []
        left_length = []
        for file in get_files(tb):
            with open(file, 'r') as f:
                words = [line.split('\t') for line in f.readlines() if line[0].isdigit()]
                words = [word for word in words if word[0].isdigit()]
                            
            for word in words:
                idx = int(word[0])
                head = int(word[6])
                
                if head != 0: # it is not root
                    if idx < head:
                        left += 1
                        left_length.append(head-idx)

                    elif idx > head:
                        right += 1
                        right_lenght.append(idx-head)
                    
                    abs_length.append(abs(head-idx))
        
        avg_l = left/len(abs_length) * 100
        avg_r = right/len(abs_length) * 100
        avg_len = sum(abs_length)/len(abs_length)
        avg_r_len = sum(right_lenght)/len(right_lenght)
        avg_l_len = sum(left_length)/len(left_length)
        std_len = np.std(abs_length)
        std_r_len = np.std(right_lenght)             
        std_l_len = np.std(left_length)


        df = df.append(
            {
                'treebank': tb,
                '% l': avg_l,
                '% r': avg_r,
                'avg_len': avg_len,
                'avg_r_len': avg_r_len,
                'avg_l_len': avg_l_len,
                'std_len': std_len,
                'std_r_len': std_r_len,
                'std_l_len': std_l_len
            }, ignore_index=True
        )
        print(f'Treebank: {tb}')
        print(f'Left: {avg_l:.2f}%')
        print(f'Right: {avg_r:.2f}%')
        print(f'Average length: {avg_len:.2f}')
        print(f'Average right length: {avg_r_len:.2f}')
        print(f'Average left length: {avg_l_len:.2f}')
        print(f'Std length: {std_len:.2f}')
        print(f'Std right length: {std_r_len:.2f}')
        print(f'Std left length: {std_l_len:.2f}')
        print()
        
    return df


class CoNLLu:
    def __init__(self, file):
        with open(file, 'r') as f:
            self.text = f.read()
        
        self.lines = [line for line in self.text.split('\n') if line != '']
        self.sentences = [sentence for sentence in self.text.split('\n\n') if sentence != '']
        self.attributes = self.get_attributes()

        self.sentence_length = self.sentence_length_dist()

    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        return self.sentences[idx]
    
    def get_attributes(self):
        """
        Return a list of dictionaries with the information of each sentence
        """
        self.attributes = {'idx': [], 'word': [], 'lemma': [], 'upos': [], 'xpos': [], 'feats': [], 'head': [], 'deprel': [], 'deps': [], 'misc': [], 'arc': []}
        self.constituency = []
        for line in self.lines:
            if line[0].isdigit():
                line = line.split('\t')
                self.attributes['idx'].append(line[0])
                self.attributes['word'].append(line[1])
                self.attributes['lemma'].append(line[2])
                self.attributes['upos'].append(line[3])
                self.attributes['xpos'].append(line[4])
                self.attributes['feats'].append(line[5])
                self.attributes['head'].append(line[6])
                self.attributes['deprel'].append(line[7])
                self.attributes['deps'].append(line[8])
                self.attributes['misc'].append(line[9])
                self.attributes['arc'].append(int(line[6]) - int(line[0])) # arc length

            elif line.startswith('# constituency = '):
                self.constituency.append(line[16:-1])
            
        return self.attributes

    def remove_outliers(self, data, threshold=1):
        """
        Remove values that appear less than threshold times
        """
        freq_dict = {}
        for x in data:
            freq_dict[x] = freq_dict.get(x, 0) + 1
        return [x for x in data if freq_dict[x] > threshold]
    
    def sentence_length_dist(self):
        return [len(sentence.split('\n')) for sentence in self.sentences]

    def avg_sentence_length(self):
        return sum(self.sentence_length) / len(self.sentences)


