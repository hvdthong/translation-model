def load_data():
    with open('small_vocab_en', "r") as f:
        data1 = f.read()
    with open('small_vocab_fr', "r") as f:
        data2 = f.read()
    return data1.split('\n'), data2.split('\n')


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


def get_value(items_tuple):
    return items_tuple[1]


if __name__ == '__main__':
    english_sentences, french_sentences = load_data()
    print(len(english_sentences), len(french_sentences))
    print("The longest english sentence in our dataset is:", longest_sentences(sentences=english_sentences))
    print("The longest french sentence in our dataset is:", longest_sentences(sentences=french_sentences))

    print('Number of unique English words:', len(word_dictionary(sentences=english_sentences)))
    print('Number of unique France words:', len(word_dictionary(sentences=french_sentences)))

    # Sort the word counts to see what words or most/least common
    en_word_count = word_dictionary(sentences=english_sentences)
    sorted_en_words = sorted(en_word_count.items(), key=get_value, reverse=True)
    print(sorted_en_words[:10])
