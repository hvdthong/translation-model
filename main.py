from data_prepared import load_embedding
from data_loader import French2EnglishDataset
from data_prepared import word_index, load_large_data, longest_sentences

if __name__ == '__main__':
    # en_words, en_vectors = load_embedding(language='en')
    # en_word2idx = word_index(word_count=list(en_words))
    # exit()
    # print(en_words.shape, en_vectors.shape, len(en_words))
    # fr_words, fr_vectors = load_embedding(language='fr')
    # fr_word2idx = word_index(word_count=list(fr_words))
    # print(fr_words.shape, fr_vectors.shape, len(en_words))

    english_sentences, french_sentences = load_large_data()
    max_en_sents, max_fr_sents = longest_sentences(sentences=english_sentences), longest_sentences(
        sentences=french_sentences)
    print(max_en_sents, max_fr_sents)

    exit()
    french_english_dataset = French2EnglishDataset(french_sentences,
                                                   english_sentences,
                                                   fr_word2idx,
                                                   en_word2idx,
                                                   seq_length=seq_length)
