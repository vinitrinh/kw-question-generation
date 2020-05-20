"""
Embedding Distributors
"""

import numpy as np
from abc import ABC, abstractmethod

import tensorflow as tf
import tensorflow_hub as hub
import sent2vec


class EmbeddingDistributor(ABC):
    """
    Abstract class in charge of providing the embeddings of piece of texts
    """
    @abstractmethod
    def get_tokenized_sents_embeddings(self, sents):
        """
        Generate a numpy ndarray with the embedding of each element of sent in each row
        :param sents: list of string (sentences/phrases)
        :return: ndarray with shape (len(sents), dimension of embeddings)
        """
        pass


class sent2vec_enc(EmbeddingDistributor):
    """
    Concrete class of @EmbeddingDistributor using a local installation of sent2vec
    https://github.com/epfml/sent2vec
    
    """

    def __init__(self, fasttext_model):
        self.model = sent2vec.Sent2vecModel()
        self.model.load_model(fasttext_model)

    def get_tokenized_sents_embeddings(self, sents):
        """
        Generate a numpy ndarray with the embedding of each element of sent in each row

        Args:
        ----
            sents: list of string (sentences/phrases)
        Return:
        -------
            ndarray with shape (len(sents), dimension of embeddings)
        """
        for sent in sents:
            if '\n' in sent:
                raise RuntimeError('New line is not allowed inside a sentence')

        return self.model.embed_sentences(sents)


class use_enc(EmbeddingDistributor):
    """
    Concrete class of @EmbeddingDistributor with USE
    """

    def __init__(self):
        self.model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

    def get_tokenized_sents_embeddings(self, sents):
        """
        Generate a numpy ndarray with the embedding of each element of sent in each row

        Args:
        ----
            sents: list of string (sentences/phrases)
        Return:
        -------
            ndarray with shape (len(sents), dimension of embeddings)
        """
        for sent in sents:
            if '\n' in sent:
                raise RuntimeError('New line is not allowed inside a sentence')

        return np.array( self.model(sents) )