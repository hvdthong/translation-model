def load_data():
    with open('small_vocab_en', "r") as f:
        data1 = f.read()
    with open('small_vocab_fr', "r") as f:
        data2 = f.read()
    return data1.split('\n'), data2.split('\n')


if __name__ == '__main__':
    english_sents, french_sents = load_data()
    print(len(english_sents), len(french_sents))
