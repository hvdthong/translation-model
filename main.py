def load_data():
    with open('small_vocab_en', "r") as f:
        data1 = f.read()
    with open('small_vocab_fr', "r") as f:
        data2 = f.read()
    return data1.split('\n'), data2.split('\n')


if __name__ == '__main__':
    english_sentences, french_sentences = load_data()
    print(len(english_sentences), len(french_sentences))

    print('Number of English sentences:', len(english_sentences),
          '\nNumber of French sentences:', len(french_sentences), '\n')
    print('Example/Target pair:\n')
    print('  ' + english_sentences[2])
    print('  ' + french_sentences[2])
