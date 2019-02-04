from data_prepared import load_embedding
from data_loader import French2EnglishDataset
from data_prepared import word_index, load_large_data, longest_sentences, word_dictionary
from torch.utils.data import DataLoader

if __name__ == '__main__':
    english_sentences, french_sentences = load_large_data()

    en_word_count = word_dictionary(sentences=english_sentences)
    en_word2idx = word_index(word_count=en_word_count)

    fr_word_count = word_dictionary(sentences=french_sentences)
    fr_word2idx = word_index(word_count=fr_word_count)

    max_en_sents, max_fr_sents = longest_sentences(sentences=english_sentences), longest_sentences(
        sentences=french_sentences)
    seq_length = max(max_en_sents, max_fr_sents) + 1
    french_english_dataset = French2EnglishDataset(french_sentences,
                                                   english_sentences,
                                                   fr_word_count,
                                                   en_word_count,
                                                   fr_word2idx,
                                                   en_word2idx,
                                                   seq_length=seq_length)
    # test_sample = french_english_dataset.__getitem__(idx=-10)  # get 10th to last item in dataset
    # print('Input example:')
    # print('Sentence:', test_sample['french_sentence'])
    # print('Tensor:', test_sample['french_tensor'])
    #
    # print('\nTarget example:')
    # print('Sentence:', test_sample['english_sentence'])
    # print('Tensor:', test_sample['english_tensor'])

    # Build dataloader to check how the batching works
    dataloader = DataLoader(french_english_dataset, batch_size=5,
                            shuffle=True, num_workers=4)
    # Prints out 10 batches from the dataloader
    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched['french_tensor'].shape,
              sample_batched['english_tensor'].shape)
        exit()



