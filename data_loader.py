import torch
from torch.utils.data import Dataset


class French2EnglishDataset(Dataset):
    '''
            French and associated English sentences.
        '''

    def __init__(self, fr_sentences, en_sentences, fr_word_count, en_word_count, fr_word2idx, en_word2idx, seq_length):
        self.fr_sentences = fr_sentences
        self.en_sentences = en_sentences
        self.fr_word_count = fr_word_count
        self.en_word_count = en_word_count
        self.fr_word2idx = fr_word2idx
        self.en_word2idx = en_word2idx
        self.seq_length = seq_length
        self.unk_en = set()
        self.unk_fr = set()

    def __len__(self):
        return len(self.fr_sentences)

    def __getitem__(self, idx):
        '''
            Returns a pair of tensors containing word indices
            for the specified sentence pair in the dataset.
        '''

        # init torch tensors, note that 0 is the padding index
        french_tensor = torch.zeros(self.seq_length, dtype=torch.long)
        english_tensor = torch.zeros(self.seq_length, dtype=torch.long)

        # Get sentence pair
        french_sentence = self.fr_sentences[idx].split()
        english_sentence = self.en_sentences[idx].split()

        # Add <EOS> tags
        french_sentence.append('</s>')
        english_sentence.append('</s>')

        # Load word indices
        for i, word in enumerate(french_sentence):
            if word in self.fr_word2idx and self.fr_word_count[word] > 5:
                french_tensor[i] = self.fr_word2idx[word]
            else:
                french_tensor[i] = self.fr_word2idx['<unk>']
                self.unk_fr.add(word)

        for i, word in enumerate(english_sentence):
            if word in self.en_word2idx and self.en_word_count[word] > 5:
                english_tensor[i] = self.en_word2idx[word]
            else:
                english_tensor[i] = self.en_word2idx['<unk>']
                self.unk_en.add(word)

        sample = {'french_tensor': french_tensor, 'french_sentence': self.fr_sentences[idx],
                  'english_tensor': english_tensor, 'english_sentence': self.en_sentences[idx]}
        return sample
