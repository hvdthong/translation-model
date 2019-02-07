from data_prepared import load_embedding
from data_loader import French2EnglishDataset
from data_prepared import word_index, load_large_data, longest_sentences, word_dictionary, load_small_data
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import random
import matplotlib.pyplot as plt
import numpy as np


class EncoderBiLSTM(nn.Module):
    def __init__(self, hidden_size, pretrained_embeddings):
        super(EncoderBiLSTM, self).__init__()

        # Model Parameters
        self.hidden_size = hidden_size
        self.embedding_dim = pretrained_embeddings.shape[1]
        self.vocab_size = pretrained_embeddings.shape[0]
        self.num_layers = 2
        self.dropout = 0.1 if self.num_layers > 1 else 0
        self.bidirectional = True

        # Construct the layers
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)

        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))  # Load the pretrained embeddings
        self.embedding.weight.requires_grad = False  # Freeze embedding layer

        self.lstm = nn.LSTM(self.embedding_dim,
                            self.hidden_size,
                            self.num_layers,
                            batch_first=True,
                            dropout=self.dropout,
                            bidirectional=self.bidirectional)

        # Initialize hidden to hidden weights in LSTM to the Identity matrix
        # This improves training and prevents exploding gradients
        # PyTorch LSTM has the 4 different hidden to hidden weights stacked in one matrix
        identity_init = torch.eye(self.hidden_size)
        self.lstm.weight_hh_l0.data.copy_(torch.cat([identity_init] * 4, dim=0))
        self.lstm.weight_hh_l0_reverse.data.copy_(torch.cat([identity_init] * 4, dim=0))
        self.lstm.weight_hh_l1.data.copy_(torch.cat([identity_init] * 4, dim=0))
        self.lstm.weight_hh_l1_reverse.data.copy_(torch.cat([identity_init] * 4, dim=0))

    def forward(self, input, hidden):
        embedded = self.embedding(input)
        output = self.lstm(embedded, hidden)
        return output

    def initHidden(self, batch_size):
        hidden_state = torch.zeros(self.num_layers * (2 if self.bidirectional else 1),
                                   batch_size,
                                   self.hidden_size,
                                   device=device)

        cell_state = torch.zeros(self.num_layers * (2 if self.bidirectional else 1),
                                 batch_size,
                                 self.hidden_size,
                                 device=device)

        return (hidden_state, cell_state)


class EncoderBiGRU(nn.Module):
    def __init__(self, hidden_size, pretrained_embeddings):
        super(EncoderBiGRU, self).__init__()

        # Model parameters
        self.hidden_size = hidden_size
        self.embedding_dim = pretrained_embeddings.shape[1]
        self.vocab_size = pretrained_embeddings.shape[0]
        self.num_layers = 2
        self.dropout = 0.1 if self.num_layers > 1 else 0
        self.bidirectional = True

        # Construct the layers
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
        self.embedding.weight.requires_grad = False

        self.gru = nn.GRU(self.embedding_dim,
                          self.hidden_size,
                          self.num_layers,
                          batch_first=True,
                          dropout=self.dropout,
                          bidirectional=self.bidirectional)

        # Initialize hidden to hidden weights in GRU to the Identity matrix
        # PyTorch GRU has 3 different hidden to hidden weights stacked in one matrix
        identity_init = torch.eye(self.hidden_size)
        self.gru.weight_hh_l0.data.copy_(torch.cat([identity_init] * 3, dim=0))
        self.gru.weight_hh_l0_reverse.data.copy_(torch.cat([identity_init] * 3, dim=0))
        self.gru.weight_hh_l1.data.copy_(torch.cat([identity_init] * 3, dim=0))
        self.gru.weight_hh_l1_reverse.data.copy_(torch.cat([identity_init] * 3, dim=0))

    def forward(self, input, hidden):
        embedded = self.embedding(input)
        output = self.gru(embedded, hidden)
        return output

    def initHidden(self, batch_size):
        hidden_state = torch.zeros(self.num_layers * (2 if self.bidirectional else 1),
                                   batch_size,
                                   self.hidden_size,
                                   device=device)

        return hidden_state


class AttnDecoderLSTM(nn.Module):
    def __init__(self, decoder_hidden_size, pretrained_embeddings, seq_length):
        super(AttnDecoderLSTM, self).__init__()
        # Embedding parameters
        self.embedding_dim = pretrained_embeddings.shape[1]
        self.output_vocab_size = pretrained_embeddings.shape[0]

        # LSTM parameters
        self.decoder_hidden_size = decoder_hidden_size
        self.num_layers = 2  # Potentially add more layers to LSTM later
        self.dropout = 0.1 if self.num_layers > 1 else 0  # Potentially add dropout later

        # Attention parameters
        self.seq_length = seq_length
        self.encoder_hidden_dim = 2 * decoder_hidden_size

        # Construct embedding layer for output language
        self.embedding = nn.Embedding(self.output_vocab_size, self.embedding_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
        self.embedding.weight.requires_grad = False  # we don't want to train the embedding weights

        # Construct layer that calculates attentional weights
        self.attn = nn.Linear((self.decoder_hidden_size + self.embedding_dim), self.seq_length)

        # Construct layer that compresses the combined matrix of the input embeddings
        # and the encoder inputs after attention has been applied
        self.attn_with_input = nn.Linear(self.embedding_dim + self.encoder_hidden_dim, self.embedding_dim)

        # LSTM for Decoder
        self.lstm = nn.LSTM(self.embedding_dim,
                            self.decoder_hidden_size,
                            self.num_layers,
                            dropout=self.dropout)

        # Initialize hidden to hidden weights in LSTM to the Identity matrix
        # PyTorch LSTM has 4 different hidden to hidden weights stacked in one matrix
        identity_init = torch.eye(self.decoder_hidden_size)
        self.lstm.weight_hh_l0.data.copy_(torch.cat([identity_init] * 4, dim=0))
        self.lstm.weight_hh_l1.data.copy_(torch.cat([identity_init] * 4, dim=0))

        # Output layer
        self.out = nn.Linear(self.decoder_hidden_size, self.output_vocab_size)

    def forward(self, input, hidden, encoder_output):
        # Input word indices, should have dim(1, batch_size), output will be (1, batch_size, embedding_dim)
        embedded = self.embedding(input)

        # Calculate Attention weights
        attn_weights = F.softmax(self.attn(torch.cat((hidden[0][1], embedded[0]), 1)), dim=1)
        attn_weights = attn_weights.unsqueeze(1)  # Add dimension for batch matrix multiplication

        # Apply Attention weights
        attn_applied = torch.bmm(attn_weights, encoder_output)
        attn_applied = attn_applied.squeeze(1)  # Remove extra dimension, dim are now (batch_size, encoder_hidden_size)

        # Prepare LSTM input tensor
        attn_combined = torch.cat((embedded[0], attn_applied), 1)  # Combine embedding input and attn_applied,
        lstm_input = F.relu(self.attn_with_input(attn_combined))  # pass through fully connected with ReLU
        lstm_input = lstm_input.unsqueeze(0)  # Add seq dimension so tensor has expected dimensions for lstm

        output, hidden = self.lstm(lstm_input, hidden)  # Output dim = (1, batch_size, decoder_hidden_size)
        output = F.log_softmax(self.out(output[0]), dim=1)  # softmax over all words in vocab

        return output, hidden, attn_weights


class AttnDecoderGRU(nn.Module):
    def __init__(self, decoder_hidden_size, pretrained_embeddings, seq_length):
        super(AttnDecoderGRU, self).__init__()
        # Embedding parameters
        self.embedding_dim = pretrained_embeddings.shape[1]
        self.output_vocab_size = pretrained_embeddings.shape[0]

        # GRU parameters
        self.decoder_hidden_size = decoder_hidden_size
        self.num_layers = 2  # Potentially add more layers to LSTM later
        self.dropout = 0.1 if self.num_layers > 1 else 0  # Potentially add dropout later

        # Attention parameters
        self.seq_length = seq_length
        self.encoder_hidden_dim = 2 * decoder_hidden_size

        # Construct embedding layer for output language
        self.embedding = nn.Embedding(self.output_vocab_size, self.embedding_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
        self.embedding.weight.requires_grad = False  # we don't want to train the embedding weights

        # Construct layer that calculates attentional weights
        self.attn = nn.Linear(self.decoder_hidden_size + self.embedding_dim, self.seq_length)

        # Construct layer that compresses the combined matrix of the input embeddings
        # and the encoder inputs after attention has been applied
        self.attn_with_input = nn.Linear(self.embedding_dim + self.encoder_hidden_dim, self.embedding_dim)

        # gru for Decoder
        self.gru = nn.GRU(self.embedding_dim,
                          self.decoder_hidden_size,
                          self.num_layers,
                          dropout=self.dropout)

        # Initialize hidden to hidden weights in GRU to the Identity matrix
        # PyTorch GRU has 3 different hidden to hidden weights stacked in one matrix
        identity_init = torch.eye(self.decoder_hidden_size)
        self.gru.weight_hh_l0.data.copy_(torch.cat([identity_init] * 3, dim=0))
        self.gru.weight_hh_l1.data.copy_(torch.cat([identity_init] * 3, dim=0))

        # Output layer
        self.out = nn.Linear(self.decoder_hidden_size, self.output_vocab_size)

    def forward(self, input, hidden, encoder_output):
        # Input word indices, should have dim(1, batch_size), output will be (1, batch_size, embedding_dim)
        embedded = self.embedding(input)

        # Calculate Attention weights
        attn_weights = F.softmax(self.attn(torch.cat((hidden[0], embedded[0]), 1)), dim=1)
        attn_weights = attn_weights.unsqueeze(1)  # Add dimension for batch matrix multiplication

        # Apply Attention weights
        attn_applied = torch.bmm(attn_weights, encoder_output)
        attn_applied = attn_applied.squeeze(1)  # Remove extra dimension, dim are now (batch_size, encoder_hidden_size)

        # Prepare GRU input tensor

        attn_combined = torch.cat((embedded[0], attn_applied), 1)  # Combine embedding input and attn_applied,
        gru_input = F.relu(self.attn_with_input(attn_combined))  # pass through fully connected with ReLU
        gru_input = gru_input.unsqueeze(0)  # Add seq dimension so tensor has expected dimensions for lstm

        output, hidden = self.gru(gru_input, hidden)  # Output dim = (1, batch_size, decoder_hidden_size)
        output = F.log_softmax(self.out(output[0]), dim=1)  # softmax over all words in vocab

        return output, hidden, attn_weights


def train(input_tensor, target_tensor, encoder, decoder,
          encoder_optimizer, decoder_optimizer, criterion):
    # Initialize encoder hidden state
    encoder_hidden = encoder.initHidden(input_tensor.shape[0])

    # clear the gradients in the optimizers
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # run forward pass through encoder on entire sequence
    encoder_output, encoder_hidden = encoder.forward(input_tensor, encoder_hidden)

    # Initialize decoder input(Start of Sentence tag) and hidden state from encoder
    decoder_input = torch.tensor([en_word2idx['<s>']] * input_tensor.shape[0], dtype=torch.long,
                                 device=device).unsqueeze(0)

    # Use correct initial hidden state dimensions depending on type of RNN
    try:
        encoder.lstm
        decoder_hidden = (encoder_hidden[0][1::2].contiguous(), encoder_hidden[1][1::2].contiguous())
    except AttributeError:
        decoder_hidden = encoder_hidden[1::2].contiguous()

    # Initialize loss
    loss = 0

    # Implement teacher forcing
    use_teacher_forcing = True if random.random() < 0.5 else False

    if use_teacher_forcing:
        # Step through target output sequence
        for di in range(seq_length):
            output, decoder_hidden, attn_weights = decoder(decoder_input,
                                                           decoder_hidden,
                                                           encoder_output)

            # Feed target as input to next item in the sequence
            decoder_input = target_tensor[di].unsqueeze(0)
            loss += criterion(output, target_tensor[di])
    else:
        # Step through target output sequence
        for di in range(seq_length):
            # Forward pass through decoder
            output, decoder_hidden, attn_weights = decoder(decoder_input,
                                                           decoder_hidden,
                                                           encoder_output)

            # Feed output as input to next item in the sequence
            decoder_input = output.topk(1)[1].view(1, -1).detach()

            # Calculate loss
            loss += criterion(output, target_tensor[di])

    # Compute the gradients
    loss.backward()

    # Clip the gradients
    nn.utils.clip_grad_norm_(encoder.parameters(), 25)
    nn.utils.clip_grad_norm_(decoder.parameters(), 25)

    # Update the weights
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item()


def trainIters(encoder, decoder, dataloader, epochs, print_every_n_batches=10, learning_rate=0.01):
    # keep track of losses
    plot_losses = []

    # Initialize Encoder Optimizer
    encoder_parameters = filter(lambda p: p.requires_grad, encoder.parameters())
    encoder_optimizer = optim.Adam(encoder_parameters, lr=learning_rate)

    # Initialize Decoder Optimizer
    decoder_parameters = filter(lambda p: p.requires_grad, decoder.parameters())
    decoder_optimizer = optim.Adam(decoder_parameters, lr=learning_rate)

    # Specify loss function, ignore the <pad> token index so it does not contribute to loss.
    criterion = nn.NLLLoss(ignore_index=0)

    # Cycle through epochs
    for epoch in range(epochs):
        loss_avg = 0
        print(f'Epoch {epoch + 1}/{epochs}')
        # Cycle through batches
        for i, batch in enumerate(dataloader):

            input_tensor = batch['french_tensor'].to(device)
            target_tensor = batch['english_tensor'].transpose(1, 0).to(device)

            loss = train(input_tensor, target_tensor, encoder, decoder,
                         encoder_optimizer, decoder_optimizer, criterion)

            loss_avg += loss
            if i % print_every_n_batches == 0 and i != 0:
                loss_avg /= print_every_n_batches
                print(f'After {i} batches, average loss/{print_every_n_batches} batches: {loss_avg}')
                plot_losses.append(loss)
                loss_avg = 0
    return plot_losses


def get_batch(dataloader):
    for batch in dataloader:
        return batch


def evaluate(input_tensor, encoder, decoder):
    with torch.no_grad():
        encoder_hidden = encoder.initHidden(1)
        encoder.eval()
        decoder.eval()

        encoder_output, encoder_hidden = encoder(input_tensor.to(device), encoder_hidden)

        decoder_input = torch.tensor([fr_word2idx['<s>']] * input_tensor.shape[0], dtype=torch.long,
                                     device=device).unsqueeze(0)
        try:
            encoder.lstm
            decoder_hidden = (encoder_hidden[0][1::2].contiguous(), encoder_hidden[1][1::2].contiguous())
        except AttributeError:
            decoder_hidden = encoder_hidden[1::2].contiguous()

        output_list = []
        attn_weight_list = np.zeros((seq_length, seq_length))
        for di in range(seq_length):
            output, decoder_hidden, attn_weights = decoder(decoder_input,
                                                           decoder_hidden,
                                                           encoder_output)

            decoder_input = output.topk(1)[1].detach()
            output_list.append(output.topk(1)[1])
            word = en_idx2word[output.topk(1)[1].item()]

            attn_weight_list[di] += attn_weights[0, 0, :].cpu().numpy()
        return output_list, attn_weight_list


if __name__ == '__main__':
    # Use gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    english_sentences, french_sentences = load_large_data()
    english_sentences, french_sentences = english_sentences[:1000], french_sentences[:1000]

    en_word_count = word_dictionary(sentences=english_sentences)
    en_word2idx = word_index(word_count=en_word_count)

    fr_word_count = word_dictionary(sentences=french_sentences)
    fr_word2idx = word_index(word_count=fr_word_count)

    max_en_sents, max_fr_sents = longest_sentences(sentences=english_sentences), longest_sentences(
        sentences=french_sentences)
    seq_length = max(max_en_sents, max_fr_sents) + 1
    print('Max seq length %i' % (seq_length))
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

    en_words, en_vectors = load_embedding(language='en')
    print(en_words.shape, en_vectors.shape)
    fr_words, fr_vectors = load_embedding(language='fr')
    print(fr_words.shape, fr_vectors.shape)

    # Build dataloader to check how the batching works
    dataloader = DataLoader(french_english_dataset, batch_size=5, shuffle=True)
    # Prints out 10 batches from the dataloader
    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched['french_tensor'].shape,
              sample_batched['english_tensor'].shape)
        break

    for i in dataloader:
        batch = i
        break

    for i in range(5):
        print('French Sentence:', batch['french_sentence'][i])
        print('English Sentence:', batch['english_sentence'][i], '\n')

    # Test the encoder on a sample input, input tensor has dimensions (batch_size, seq_length)
    # all the variable have test_ in front of them so they don't reassign variables needed later on with the real models

    test_batch_size = 1
    test_seq_length = 3
    test_hidden_size = 5
    test_encoder = EncoderBiLSTM(test_hidden_size, fr_vectors).to(device)
    test_hidden = test_encoder.initHidden(test_batch_size)

    # Create an input tensor of random indices
    test_inputs = torch.randint(0, 50, (test_batch_size, test_seq_length), dtype=torch.long, device=device)

    test_encoder_output, test_encoder_hidden = test_encoder.forward(test_inputs, test_hidden)

    print("The final output of the BiLSTM Encoder on our test input is: ", test_encoder_output.shape)

    print('\n\nEncoder output tensor: \n\n', test_encoder_output)

    # print(test_encoder_hidden[0][::2].shape)

    # test_encoder_gru = EncoderBiGRU(test_hidden_size, fr_vectors).to(device)
    # test_hidden = test_encoder_gru.initHidden(test_batch_size)
    # o, h = test_encoder_gru(test_inputs, test_hidden)
    #
    # print(o.shape)

    # Initialize attention weights to one, note the dimensions
    attn_weights = torch.ones((test_batch_size, test_seq_length), device=device)

    # Set all weights except the weights associated with the first sequence item equal to zero
    # This would represent full attention on the first word in the sequence
    attn_weights[:, 1:] = 0

    attn_weights.unsqueeze_(1)  # Add dimension for batch matrix multiplication

    # BMM(Batch Matrix Multiply) muliplies the [1 x seq_length] matrix by the [seq_length x hidden_size] matrix for
    # each batch. This produces a single vector(for each batch) of length(encoder_hidden_size) that is the weighted
    # sum of the encoder hidden vectors for each item in the sequence.

    attn_applied = torch.bmm(attn_weights, test_encoder_output)
    attn_applied.squeeze_()  # Remove extra dimension

    print('Attention weights shape: ', attn_weights.shape)
    print('Attention weights:\n', attn_weights)
    print('\nFirst sequence item in Encoder output: \n', test_encoder_output[:, 0, :])
    print('\nEncoder Output after attention is applied: \n', attn_applied)
    print('\n', attn_applied.shape)

    # Test the decoder on sample inputs to check that the dimensions of everything is correct
    test_decoder_hidden_size = 5

    test_decoder = AttnDecoderLSTM(test_decoder_hidden_size, en_vectors, test_seq_length).to(device)

    input_idx = torch.tensor([fr_word2idx['<s>']] * test_batch_size, dtype=torch.long, device=device)
    print(input_idx.shape)

    input_idx = input_idx.unsqueeze_(0)
    test_decoder_hidden = (test_encoder_hidden[0][1::2].contiguous(), test_encoder_hidden[1][1::2].contiguous())
    print(input_idx.shape)

    output, hidden, attention = test_decoder.forward(input_idx, test_decoder_hidden, test_encoder_output)
    print(output.shape)

    test_decoder_hidden[0].shape

    # Set hyperparameters and construct dataloader
    hidden_size = 256
    batch_size = 16
    dataloader = DataLoader(french_english_dataset, batch_size=batch_size,
                            shuffle=True, num_workers=4)
    print('Length of data loader: %i' % (len(dataloader.dataset)))

    # Construct encoder and decoder instances
    encoder_lstm = EncoderBiLSTM(hidden_size, fr_vectors).to(device)
    decoder_lstm = AttnDecoderLSTM(hidden_size, en_vectors, seq_length).to(device)

    encoder_gru = EncoderBiGRU(hidden_size, fr_vectors).to(device)
    decoder_gru = AttnDecoderGRU(hidden_size, en_vectors, seq_length).to(device)

    # from_scratch = False  # Set to False if you have saved weights and want to load them
    from_scratch = True # Set to False if you have saved weights and want to load them

    if not from_scratch:
        # Load weights from earlier model
        encoder_lstm_state_dict = torch.load('models/encoder_lstm.pth')
        decoder_lstm_state_dict = torch.load('models/decoder_lstm.pth')

        encoder_lstm.load_state_dict(encoder_lstm_state_dict)
        decoder_lstm.load_state_dict(decoder_lstm_state_dict)

        # Load weights from earlier model
        encoder_gru_state_dict = torch.load('models/encoder_gru.pth')
        decoder_gru_state_dict = torch.load('models/decoder_gru.pth')

        encoder_gru.load_state_dict(encoder_gru_state_dict)
        decoder_gru.load_state_dict(decoder_gru_state_dict)
    else:
        print('Training model from scratch.')

    print('Training LSTM based network.')
    learning_rate = 0.0001
    encoder_lstm.train()  # Set model to training mode
    decoder_lstm.train()  # Set model to training mode

    lstm_losses = trainIters(encoder_lstm, decoder_lstm, dataloader, epochs=50, learning_rate=learning_rate)
    np.save('data/lstm_losses.npy', lstm_losses)
    lstm_losses = np.load('data/lstm_losses.npy')

    print('Training GRU based network.')
    learning_rate = 0.0001
    encoder_gru.train()  # Set model to training mode
    decoder_gru.train()  # Set model to training mode

    gru_losses = trainIters(encoder_gru, decoder_gru, dataloader, epochs=50, learning_rate=learning_rate)
    np.save('data/gru_losses.npy', gru_losses)
    gru_losses = np.load('data/gru_losses.npy')

    plt.plot(lstm_losses)
    plt.plot(gru_losses)

    plt.title('Loss Plots for Dataset 1; Trained on 1 Epoch')
    plt.xlabel('Batches')
    plt.xticks([0, 20, 40, 60, 80], [0, 2000, 4000, 6000, 8000])
    plt.ylabel('Loss per Batch, MSE')
    plt.legend(['LSTM', 'GRU'])
    plt.show()

    # Save the model weights to continue later
    torch.save(encoder_lstm.state_dict(), 'models/encoder_lstm.pth')
    torch.save(decoder_lstm.state_dict(), 'models/decoder_lstm.pth')

    torch.save(encoder_gru.state_dict(), 'models/encoder_gru.pth')
    torch.save(decoder_gru.state_dict(), 'models/decoder_gru.pth')

    # # Build the idx to word dictionaries to convert predicted indices to words
    # en_idx2word = {k: i for i, k in en_word2idx.items()}
    # fr_idx2word = {k: i for i, k in fr_word2idx.items()}
    # batch = get_batch(dataloader)
    # num_batch = 15
    # input_tensor = batch['french_tensor'][num_batch].unsqueeze_(0)
    # output_list, attn = evaluate(input_tensor, encoder_lstm, decoder_lstm)
    # gru_output_list, gru_attn = evaluate(input_tensor, encoder_gru, decoder_gru)
    #
    # print('Input Sentence:')
    # output = ''
    # for index in input_tensor[0]:
    #     word = fr_idx2word[index.item()]
    #     if word != '</s>':
    #         output += ' ' + word
    #     else:
    #         output += ' ' + word
    #         print(output)
    #         break
    #
    # print('\nTarget Sentence:')
    # print(' ' + batch['english_sentence'][num_batch] + '</s>')
    # input_len = len(batch['french_sentence'][num_batch].split())
    #
    # print('\nLSTM model output:')
    # output = ''
    # for index in output_list:
    #     word = en_idx2word[index.item()]
    #     if word != '</s>':
    #         output += ' ' + word
    #     else:
    #         output += ' ' + word
    #         print(output)
    #         break
    #
    # fig = plt.figure()
    # plt.title('LSTM Model Attention\n\n\n\n\n')
    # ax = fig.add_subplot(111)
    # ax.matshow(attn[:len(output.split()), :input_len])
    # ax.set_xticks(np.arange(0, input_len, step=1))
    # ax.set_yticks(np.arange(0, len(output.split())))
    # ax.set_xticklabels(batch['french_sentence'][num_batch].split(), rotation=90)
    # ax.set_yticklabels(output.split() + ['</s>'])
    # # plt.show()
    # plt.savefig('lstm.jpg')
    #
    # output = ''
    # print('\nGRU model output:')
    # for index in gru_output_list:
    #     word = en_idx2word[index.item()]
    #     if word != '</s>':
    #         output += ' ' + word
    #     else:
    #         output += ' ' + word
    #         print(output)
    #         break
    #
    # fig = plt.figure()
    # plt.title('GRU Model Attention\n\n\n\n\n')
    # ax2 = fig.add_subplot(111)
    # ax2.matshow(gru_attn[:len(output.split()), :input_len])
    # ax2.set_xticks(np.arange(0, input_len, step=1))
    # ax2.set_yticks(np.arange(0, len(output.split())))
    # ax2.set_xticklabels(batch['french_sentence'][num_batch].split(), rotation=90)
    # ax2.set_yticklabels(output.split() + ['</s>'])
    # # plt.show()
    # plt.savefig('gru.jpg')