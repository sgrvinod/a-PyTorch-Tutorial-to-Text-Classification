import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence


class HierarchialAttentionNetwork(nn.Module):
    """
    The overarching Hierarchial Attention Network (HAN).
    """

    def __init__(self, n_classes, vocab_size, emb_size, word_rnn_size, sentence_rnn_size, word_rnn_layers,
                 sentence_rnn_layers, word_att_size, sentence_att_size, dropout=0.5):
        """
        :param n_classes: number of classes
        :param vocab_size: number of words in the vocabulary of the model
        :param emb_size: size of word embeddings
        :param word_rnn_size: size of (bidirectional) word-level RNN
        :param sentence_rnn_size: size of (bidirectional) sentence-level RNN
        :param word_rnn_layers: number of layers in word-level RNN
        :param sentence_rnn_layers: number of layers in sentence-level RNN
        :param word_att_size: size of word-level attention layer
        :param sentence_att_size: size of sentence-level attention layer
        :param dropout: dropout
        """
        super(HierarchialAttentionNetwork, self).__init__()

        # Sentence-level attention module (which will, in-turn, contain the word-level attention module)
        self.sentence_attention = SentenceAttention(vocab_size, emb_size, word_rnn_size, sentence_rnn_size,
                                                    word_rnn_layers, sentence_rnn_layers, word_att_size,
                                                    sentence_att_size, dropout)

        # Classifier
        self.fc = nn.Linear(2 * sentence_rnn_size, n_classes)

        self.dropout = nn.Dropout(dropout)

    def forward(self, documents, sentences_per_document, words_per_sentence):
        """
        Forward propagation.

        :param documents: encoded document-level data, a tensor of dimensions (n_documents, sent_pad_len, word_pad_len)
        :param sentences_per_document: document lengths, a tensor of dimensions (n_documents)
        :param words_per_sentence: sentence lengths, a tensor of dimensions (n_documents, sent_pad_len)
        :return: class scores, attention weights of words, attention weights of sentences
        """
        # Apply sentence-level attention module (and in turn, word-level attention module) to get document embeddings
        document_embeddings, word_alphas, sentence_alphas = self.sentence_attention(documents, sentences_per_document,
                                                                                    words_per_sentence)  # (n_documents, 2 * sentence_rnn_size), (n_documents, max_doc_len_in_batch, max_sent_len_in_batch), # (n_documents, max_doc_len_in_batch)

        # Classify
        scores = self.fc(self.dropout(document_embeddings))  # (n_documents, n_classes)

        return scores, word_alphas, sentence_alphas


class SentenceAttention(nn.Module):
    """
    The sentence-level attention module.
    """

    def __init__(self, vocab_size, emb_size, word_rnn_size, sentence_rnn_size, word_rnn_layers, sentence_rnn_layers,
                 word_att_size, sentence_att_size, dropout):
        """
        :param vocab_size: number of words in the vocabulary of the model
        :param emb_size: size of word embeddings
        :param word_rnn_size: size of (bidirectional) word-level RNN
        :param sentence_rnn_size: size of (bidirectional) sentence-level RNN
        :param word_rnn_layers: number of layers in word-level RNN
        :param sentence_rnn_layers: number of layers in sentence-level RNN
        :param word_att_size: size of word-level attention layer
        :param sentence_att_size: size of sentence-level attention layer
        :param dropout: dropout
        """
        super(SentenceAttention, self).__init__()

        # Word-level attention module
        self.word_attention = WordAttention(vocab_size, emb_size, word_rnn_size, word_rnn_layers, word_att_size,
                                            dropout)

        # Bidirectional sentence-level RNN
        self.sentence_rnn = nn.GRU(2 * word_rnn_size, sentence_rnn_size, num_layers=sentence_rnn_layers,
                                   bidirectional=True, dropout=dropout, batch_first=True)

        # Sentence-level attention network
        self.sentence_attention = nn.Linear(2 * sentence_rnn_size, sentence_att_size)

        # Sentence context vector to take dot-product with
        self.sentence_context_vector = nn.Linear(sentence_att_size, 1,
                                                 bias=False)  # this performs a dot product with the linear layer's 1D parameter vector, which is the sentence context vector
        # You could also do this with:
        # self.sentence_context_vector = nn.Parameter(torch.FloatTensor(1, sentence_att_size))
        # self.sentence_context_vector.data.uniform_(-0.1, 0.1)
        # And then take the dot-product

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, documents, sentences_per_document, words_per_sentence):
        """
        Forward propagation.

        :param documents: encoded document-level data, a tensor of dimensions (n_documents, sent_pad_len, word_pad_len)
        :param sentences_per_document: document lengths, a tensor of dimensions (n_documents)
        :param words_per_sentence: sentence lengths, a tensor of dimensions (n_documents, sent_pad_len)
        :return: document embeddings, attention weights of words, attention weights of sentences
        """
        # Sort documents by decreasing document lengths (SORTING #1)
        sentences_per_document, doc_sort_ind = sentences_per_document.sort(dim=0, descending=True)
        documents = documents[doc_sort_ind]  # (n_documents, sent_pad_len, word_pad_len)
        words_per_sentence = words_per_sentence[doc_sort_ind]  # (n_documents, sent_pad_len)

        # Re-arrange as sentences by removing pad-sentences (DOCUMENTS -> SENTENCES)
        sentences, bs = pack_padded_sequence(documents,
                                             lengths=sentences_per_document.tolist(),
                                             batch_first=True)  # (n_sentences, word_pad_len), bs is the effective batch size at each sentence-timestep

        # Re-arrange sentence lengths in the same way (DOCUMENTS -> SENTENCES)
        words_per_sentence, _ = pack_padded_sequence(words_per_sentence,
                                                     lengths=sentences_per_document.tolist(),
                                                     batch_first=True)  # (n_sentences), '_' is the same as 'bs' in the earlier step

        # Find sentence embeddings by applying the word-level attention module
        sentences, word_alphas = self.word_attention(sentences,
                                                     words_per_sentence)  # (n_sentences, 2 * word_rnn_size), (n_sentences, max_sent_len_in_batch)
        sentences = self.dropout(sentences)

        # Apply the sentence-level RNN over the sentence embeddings (PyTorch automatically applies it on the packed_sequence using the effective batch_size)
        (sentences, _), _ = self.sentence_rnn(
            PackedSequence(sentences, bs))  # (n_sentences, 2 * sentence_rnn_size), (max(sent_lens))

        # Find attention vectors by applying the attention linear layer
        att_s = self.sentence_attention(sentences)  # (n_sentences, att_size)
        att_s = F.tanh(att_s)  # (n_sentences, att_size)
        # Take the dot-product of the attention vectors with the context vector (i.e. parameter of linear layer)
        att_s = self.sentence_context_vector(att_s).squeeze(1)  # (n_sentences)

        # Compute softmax over the dot-product manually
        # Manually because they have to be computed only over sentences in the same document

        # First, take the exponent
        max_value = att_s.max()  # scalar, for numerical stability during exponent calculation
        att_s = torch.exp(att_s - max_value)  # (n_sentences)

        # Re-arrange as documents by re-padding with 0s (SENTENCES -> DOCUMENTS)
        att_s, _ = pad_packed_sequence(PackedSequence(att_s, bs),
                                       batch_first=True)  # (n_documents, max_doc_len_in_batch)

        # Calculate softmax values
        sentence_alphas = att_s / torch.sum(att_s, dim=1, keepdim=True)  # (n_documents, max_doc_len_in_batch)

        # Similarly re-arrange sentence-level RNN outputs as documents by re-padding with 0s (SENTENCES -> DOCUMENTS)
        documents, _ = pad_packed_sequence(PackedSequence(sentences, bs),
                                           batch_first=True)  # (n_documents, max_doc_len_in_batch, 2 * sentence_rnn_size)

        # Find document embeddings
        documents = documents * sentence_alphas.unsqueeze(
            2)  # (n_documents, max_doc_len_in_batch, 2 * sentence_rnn_size)
        documents = documents.sum(dim=1)  # (n_documents, 2 * sentence_rnn_size)

        # Also re-arrange word_alphas (SENTENCES -> DOCUMENTS)
        word_alphas, _ = pad_packed_sequence(PackedSequence(word_alphas, bs),
                                             batch_first=True)  # (n_documents, max_doc_len_in_batch, max_sent_len_in_batch)

        # Unsort documents into the original order (INVERSE OF SORTING #1)
        _, doc_unsort_ind = doc_sort_ind.sort(dim=0, descending=False)  # (n_documents)
        documents = documents[doc_unsort_ind]  # (n_documents, 2 * sentence_rnn_size)
        sentence_alphas = sentence_alphas[doc_unsort_ind]  # (n_documents, max_doc_len_in_batch)
        word_alphas = word_alphas[doc_unsort_ind]  # (n_documents, max_doc_len_in_batch, max_sent_len_in_batch)

        return documents, word_alphas, sentence_alphas


class WordAttention(nn.Module):
    """
    The word-level attention module.
    """

    def __init__(self, vocab_size, emb_size, word_rnn_size, word_rnn_layers, word_att_size, dropout):
        """
        :param vocab_size: number of words in the vocabulary of the model
        :param emb_size: size of word embeddings
        :param word_rnn_size: size of (bidirectional) word-level RNN
        :param word_rnn_layers: number of layers in word-level RNN
        :param word_att_size: size of word-level attention layer
        :param dropout: dropout
        """
        super(WordAttention, self).__init__()

        # Embeddings (look-up) layer
        self.embeddings = nn.Embedding(vocab_size, emb_size)

        # Bidirectional word-level RNN
        self.word_rnn = nn.GRU(emb_size, word_rnn_size, num_layers=word_rnn_layers, bidirectional=True,
                               dropout=dropout, batch_first=True)

        # Word-level attention network
        self.word_attention = nn.Linear(2 * word_rnn_size, word_att_size)

        # Word context vector to take dot-product with
        self.word_context_vector = nn.Linear(word_att_size, 1, bias=False)
        # You could also do this with:
        # self.word_context_vector = nn.Parameter(torch.FloatTensor(1, word_att_size))
        # self.word_context_vector.data.uniform_(-0.1, 0.1)
        # And then take the dot-product

        self.dropout = nn.Dropout(dropout)

    def init_embeddings(self, embeddings):
        """
        Initialized embedding layer with pre-computed embeddings.

        :param embeddings: pre-computed embeddings
        """
        self.embeddings.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune=False):
        """
        Allow fine-tuning of embedding layer? (Only makes sense to not-allow if using pre-trained embeddings).

        :param fine_tune: allow?
        """
        for p in self.embeddings.parameters():
            p.requires_grad = fine_tune

    def forward(self, sentences, words_per_sentence):
        """
        Forward propagation.

        :param sentences: encoded sentence-level data, a tensor of dimension (n_sentences, word_pad_len, emb_size)
        :param words_per_sentence: sentence lengths, a tensor of dimension (n_sentences)
        :return: sentence embeddings, attention weights of words
        """
        # Sort sentences by decreasing sentence lengths (SORTING #2)
        words_per_sentence, sent_sort_ind = words_per_sentence.sort(dim=0, descending=True)
        sentences = sentences[sent_sort_ind]  # (n_sentences, word_pad_len, emb_size)

        # Get word embeddings, apply dropout
        sentences = self.dropout(self.embeddings(sentences))  # (n_sentences, word_pad_len, emb_size)

        # Re-arrange as words by removing pad-words (SENTENCES -> WORDS)
        words, bw = pack_padded_sequence(sentences,
                                         lengths=words_per_sentence.tolist(),
                                         batch_first=True)  # (n_words, emb_size), bw is the effective batch size at each word-timestep

        # Apply the word-level RNN over the word embeddings (PyTorch automatically applies it on the packed_sequence using the effective batch_size)
        (words, _), _ = self.word_rnn(PackedSequence(words, bw))  # (n_words, 2 * word_rnn_size), (max(sent_lens))

        # Find attention vectors by applying the attention linear layer
        att_w = self.word_attention(words)  # (n_words, att_size)
        att_w = F.tanh(att_w)  # (n_words, att_size)
        # Take the dot-product of the attention vectors with the context vector (i.e. parameter of linear layer)
        att_w = self.word_context_vector(att_w).squeeze(1)  # (n_words)

        # Compute softmax over the dot-product manually
        # Manually because they have to be computed only over words in the same sentence

        # First, take the exponent
        max_value = att_w.max()  # scalar, for numerical stability during exponent calculation
        att_w = torch.exp(att_w - max_value)  # (n_words)

        # Re-arrange as sentences by re-padding with 0s (WORDS -> SENTENCES)
        att_w, _ = pad_packed_sequence(PackedSequence(att_w, bw),
                                       batch_first=True)  # (n_sentences, max_sent_len_in_batch)

        # Calculate softmax values
        word_alphas = att_w / torch.sum(att_w, dim=1, keepdim=True)  # (n_sentences, max_sent_len_in_batch)

        # Similarly re-arrange word-level RNN outputs as sentences by re-padding with 0s (WORDS -> SENTENCES)
        sentences, _ = pad_packed_sequence(PackedSequence(words, bw),
                                           batch_first=True)  # (n_sentences, max_sent_len_in_batch, 2 * word_rnn_size)

        # Find sentence embeddings
        sentences = sentences * word_alphas.unsqueeze(2)  # (n_sentences, max_sent_len_in_batch, 2 * word_rnn_size)
        sentences = sentences.sum(dim=1)  # (n_sentences, 2 * word_rnn_size)

        # Unsort sentences into the original order (INVERSE OF SORTING #2)
        _, sent_unsort_ind = sent_sort_ind.sort(dim=0, descending=False)  # (n_sentences)
        sentences = sentences[sent_unsort_ind]  # (n_sentences, 2 * word_rnn_size)
        word_alphas = word_alphas[sent_unsort_ind]  # (n_sentences, max_sent_len_in_batch)

        return sentences, word_alphas
