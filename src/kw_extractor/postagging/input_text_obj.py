
import argparse
import os
import re
import warnings
from abc import ABC, abstractmethod

# NLTK imports
import nltk
from nltk.tag.util import tuple2str
from nltk.parse import CoreNLPParser
from nltk.stem import PorterStemmer
import stanfordnlp

from .fileIO import read_file, write_string

GRAMMAR_EN = """  NP:
        {<NN.*|JJ>*<NN.*>}  # Adjective(s)(optional) + Noun(s)"""

class InputTextObj:
    """Represent the input text in which we want to extract keyphrases"""

    def __init__(self, pos_tagged, stem=False, min_word_len=3):
        """
        :param pos_tagged: List of list : Text pos_tagged as a list of sentences
        where each sentence is a list of tuple (word, TAG).
        :param stem: If we want to apply stemming on the text.
        """
        self.min_word_len = min_word_len
        self.considered_tags = {'NN', 'NNS', 'NNP', 'NNPS', 'JJ'}
        self.pos_tagged = []
        self.filtered_pos_tagged = []
        self.isStemmed = stem

        if stem:
            stemmer = PorterStemmer()
            self.pos_tagged = [[(stemmer.stem(t[0]), t[1]) for t in sent] for sent in pos_tagged]
        else:
            self.pos_tagged = [[(t[0].lower(), t[1]) for t in sent] for sent in pos_tagged]

        temp = []
        for sent in self.pos_tagged:
            s = []
            for elem in sent:
                if len(elem[0]) < min_word_len:
                    s.append((elem[0], 'LESS'))
                else:
                    s.append(elem)
            temp.append(s)

        self.pos_tagged = temp
        self.filtered_pos_tagged = [[(t[0].lower(), t[1]) for t in sent if self.is_candidate(t)] for sent in
                                    self.pos_tagged]

    def is_candidate(self, tagged_token):
        """

        :param tagged_token: tuple (word, tag)
        :return: True if its a valid candidate word
        """
        return tagged_token[1] in self.considered_tags

    def extract_candidate_phrases(self, no_subset=False):
        """
        Based on part of speech return a list of candidate phrases

        args:
        -----
            text_obj: Input text Representation see @InputTextObj
            no_subset: if true won't put a candidate which is the subset of an other candidate
            lang: language (currently en, fr and de are supported)
            
        Return:
        ------
            list of candidate phrases (string)
        """

        """
        1. extract noun phrases using NLTK.

        Sample keyphrase_candidate:
        -------------------------
            {'carolingian dynasty',
            'england',
            'feudal doctrines',
            'france',
            'franks',
            'functional hierarchical system',
            'lineage',
            'new normanrulers',
            'normandy',
            'normans',
            'old french aristocracy',
            'rest'}
        """
        keyphrase_candidate = set()

        np_parser = nltk.RegexpParser(GRAMMAR_EN)  
        trees = np_parser.parse_sents(self.pos_tagged)  # Generator with one tree per sentence


        for tree in trees:
            for subtree in tree.subtrees(filter=lambda t: t.label() == 'NP'):  # For each nounphrase
                # Concatenate the token with a space
                keyphrase_candidate.add(' '.join(word for word, tag in subtree.leaves()))

        keyphrase_candidate = {kp for kp in keyphrase_candidate if len(kp.split()) <= 5}
        """
        2. handle overlapping keyphrases:

        sample output:
        -------------
            ['functional hierarchical system',
            'old french aristocracy',
            'carolingian dynasty',
            'feudal doctrines',
            'new normanrulers',
            'normandy',
            'normans',
            'england',
            'lineage',
            'franks',
            'france',
            'rest']
        """
        if no_subset:
            keyphrase_candidate = unique_ngram_candidates(keyphrase_candidate)
        else:
            keyphrase_candidate = list(keyphrase_candidate)

        return keyphrase_candidate

def unique_ngram_candidates(strings):
    """
    ['machine learning', 'machine', 'backward induction', 'induction', 'start'] ->
    ['backward induction', 'start', 'machine learning']
    
    Args:
    ----
        strings: List of string
    Return:
    ------
        List of string where no string is fully contained inside another string
    """
    results = []
    for s in sorted(set(strings), key=len, reverse=True):
        if not any(re.search(r'\b{}\b'.format(re.escape(s)), r) for r in results):
            results.append(s)
    return results
    
def convert(fr_or_de_tag):
    if fr_or_de_tag in {'NN', 'NNE', 'NE', 'N', 'NPP', 'NC', 'NOUN'}:
        return 'NN'
    elif fr_or_de_tag in {'ADJA', 'ADJ'}:
        return 'JJ'
    else:
        return fr_or_de_tag
