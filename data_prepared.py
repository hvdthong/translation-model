import string
import numpy as np
import os


def load_data():
    with open('./data/small_vocab_en', "r") as f:
        data1 = f.read()
    with open('./data/small_vocab_fr', "r") as f:
        data2 = f.read()
    return data1.split('\n'), data2.split('\n')


def load_large_data():
    with open('data/fra.txt', "r") as f:
        data1 = f.read()
    pairs = data1.split('\n')
    english_sentences = []
    french_sentences = []
    for i, pair in enumerate(pairs):
        pair_split = pair.split('\t')
        if len(pair_split) != 2:
            continue
        english = pair_split[0].lower()
        french = pair_split[1].lower()

        # Remove punctuation and limit sentence length
        max_sent_length = 10
        punctuation_table = english.maketrans({i: None for i in string.punctuation})
        english = english.translate(punctuation_table)
        french = french.translate(punctuation_table)
        if len(english.split()) > max_sent_length or len(french.split()) > max_sent_length:
            continue

        english_sentences.append(english)
        french_sentences.append(french)
    return english_sentences, french_sentences


def longest_sentences(sentences):
    max_length = 0
    for sentence in sentences:
        length = len(sentence.split())
        max_length = max(max_length, length)
    return max_length


def word_dictionary(sentences):
    word_count = {}
    for sent in sentences:
        for word in sent.split():
            if word in word_count:
                word_count[word] += 1
            else:
                word_count[word] = 1
    word_count['</s>'] = len(sentences)
    return word_count


def word_index(word_count):
    word2idx = {k: v + 3 for v, k in enumerate(word_count.keys())}
    return word2idx


def load_embedding(language):
    word_file, vector_file = language + '_words.npy', language + '_vectors.npy'
    if os.path.exists('data/' + word_file) and os.path.exists('data/' + vector_file):
        words = np.load('data/' + word_file)
        vectors = np.load('data/' + vector_file)
        print('Embeddings load from .npy file')
    else:
        # make a dict with the top 100,000 words
        words = ['<pad>',  # Padding Token
                 '<s>',  # Start of sentence token
                 '<unk>'  # Unknown word token
                 ]

        vectors = list(np.random.uniform(-0.1, 0.1, (3, 300)))
        vectors[0] *= 0  # make the padding vector zeros
        with open('data/cc.' + language + '.300.vec', "r", encoding="utf8") as f:
            f.readline()
            for _ in range(100000):
                vecs = f.readline()
                word = vecs.split()[0]
                vector = np.float32(vecs.split()[1:])

                # skip lines that don't have 300 dim
                if len(vector) != 300:
                    continue

                if word not in words:
                    words.append(word)
                    vectors.append(vector)
            # print(word, vector[:10])  # Last word embedding read from the file
            words = np.array(words)
            vectors = np.array(vectors)
        # Save the arrays so we don't have to load the full word embedding file
        np.save('data/' + word_file, words)
        np.save('data/' + vector_file, vectors)
    return words, vectors


def get_value(items_tuple):
    return items_tuple[1]


if __name__ == '__main__':
    # english_sentences, french_sentences = load_data()
    # english_sentences, french_sentences = load_large_data()
    # print(len(english_sentences), len(french_sentences))
    # print("The longest english sentence in our dataset is:", longest_sentences(sentences=english_sentences))
    # print("The longest french sentence in our dataset is:", longest_sentences(sentences=french_sentences))
    #
    # print('Number of unique English words:', len(word_dictionary(sentences=english_sentences)))
    # print('Number of unique France words:', len(word_dictionary(sentences=french_sentences)))
    #
    # # Sort the word counts to see what words or most/least common
    # en_word_count = word_dictionary(sentences=english_sentences)
    # sorted_en_words = sorted(en_word_count.items(), key=get_value, reverse=True)
    # print(sorted_en_words[:10])
    #
    # fr_word_count = word_dictionary(sentences=english_sentences)
    # sorted_fr_words = sorted(fr_word_count.items(), key=get_value, reverse=True)
    # print(sorted_fr_words[:10])
    #
    # fr_word2idx = word_index(word_count=fr_word_count)
    # print(len(fr_word2idx))

    en_words, en_vectors = load_embedding(language='en')
    print(en_words.shape, en_vectors.shape)
    fr_words, fr_vectors = load_embedding(language='fr')
    print(fr_words.shape, fr_vectors.shape)
    exit()

    print(len(en_words), len(en_vectors))
    print(type(en_words), type(en_vectors))
    print(en_words.shape, en_vectors.shape)

    en_word2idx = word_index(word_count=en_words)
    hemophilia_idx = en_word2idx['hemophilia']
    print('index for word hemophilia:', hemophilia_idx,
          '\nvector for word hemophilia:\n', en_vectors[hemophilia_idx][:10])

    fr_words, fr_vectors = load_embedding(language='fr')
    fr_word2idx = {word: index for index, word in enumerate(fr_words)}
    chabeuil_idx = fr_word2idx['chabeuil']
    print('index for word chabeuil:', chabeuil_idx,
          '\nvector for word chabeuil:\n', fr_vectors[chabeuil_idx][:10])
