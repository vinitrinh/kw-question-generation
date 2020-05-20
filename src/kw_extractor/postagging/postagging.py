# Copyright (c) 2017-present, Swisscom (Schweiz) AG.
# All rights reserved.
#
#Authors: Kamil Bennani-Smires, Yann Savary

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
import stanza

from .fileIO import read_file, write_string


class PosTagging(ABC):
    @abstractmethod
    def pos_tag_raw_text(self, text, as_tuple_list=True):
        """
        Tokenize and POS tag a string
        Sentence level is kept in the result :
        Either we have a list of list (for each sentence a list of tuple (word,tag))
        Or a separator [ENDSENT] if we are requesting a string by putting as_tuple_list = False

        Example :
        >>> from sentkp.preprocessing import postagger as pt
        >>> pt = postagger.PosTagger()
        >>> pt.pos_tag_raw_text('Write your python code in a .py file. Thank you.')
        [
            [('Write', 'VB'), ('your', 'PRP$'), ('python', 'NN'),
            ('code', 'NN'), ('in', 'IN'), ('a', 'DT'), ('.', '.'), ('py', 'NN'), ('file', 'NN'), ('.', '.')
            ],
            [('Thank', 'VB'), ('you', 'PRP'), ('.', '.')]
        ]

        >>> pt.pos_tag_raw_text('Write your python code in a .py file. Thank you.', as_tuple_list=False)
        'Write/VB your/PRP$ python/NN code/NN in/IN a/DT ./.[ENDSENT]py/NN file/NN ./.[ENDSENT]Thank/VB you/PRP ./.'


        >>> pt = postagger.PosTagger(separator='_')
        >>> pt.pos_tag_raw_text('Write your python code in a .py file. Thank you.', as_tuple_list=False)
        Write_VB your_PRP$ python_NN code_NN in_IN a_DT ._. py_NN file_NN ._.
        Thank_VB you_PRP ._.



        :param as_tuple_list: Return result as list of list (word,Pos_tag)
        :param text:  String to POS tag
        :return: POS Tagged string or Tuple list
        """

        pass

    def pos_tag_file(self, input_path, output_path=None):

        """
        POS Tag a file.
        Either we have a list of list (for each sentence a list of tuple (word,tag))
        Or a file with the POS tagged text

        Note : The jumpline is only for readibility purpose , when reading a tagged file we'll use again
        sent_tokenize to find the sentences boundaries.

        args:
        -----
            input_path: path of the source file
            output_path: If set write POS tagged text with separator (self.pos_tag_raw_text with as_tuple_list False)
                         If not set, return list of list of tuple (self.post_tag_raw_text with as_tuple_list = True)

        Return:
        ------
            resulting POS tagged text as a list of list of tuple or nothing if output path is set.
        
        """

        original_text = read_file(input_path)

        if output_path is not None:
            tagged_text = self.pos_tag_raw_text(original_text, as_tuple_list=False)
            # Write to the output the POS-Tagged text.
            write_string(tagged_text, output_path)
        else:
            return self.pos_tag_raw_text(original_text, as_tuple_list=True)

    def pos_tag_and_write_corpora(self, list_of_path, suffix):
        """
        POS tag a list of files
        It writes the resulting file in the same directory with the same name + suffix

        args:
        ----
            list_of_path: list containing the path (as string) of each file to POS Tag
            suffix: suffix to append at the end of the original filename for the resulting pos_tagged file.

        Sample usage:
        -------------
        >>> list_of_path = read_file(args.listing_file_path).splitlines()
        >>> print('POS Tagging and writing ', len(list_of_path), 'files')
        >>> pt.pos_tag_and_write_corpora(['/Users/user1/text1', '/Users/user1/direct/text2'] , suffix = _POS)
        
        creates
        /Users/user1/text1_POS
        /Users/user1/direct/text2_POS
        
        """
        for path in list_of_path:
            output_file_path = path + suffix
            if os.path.isfile(path):
                self.pos_tag_file(path, output_file_path)
            else:
                warnings.warn('file ' + output_file_path + 'does not exists')

class PosTaggingStanza(PosTagging):
    """
    Concrete class of PosTagging using a CoreNLP server 
    Provides a faster way to process several documents using 
    since it doesn't require to load the model each time.
    """

    def __init__(self, separator='|'):
        self.parser = stanza.Pipeline(processors = "tokenize,pos", use_gpu=False)
        self.separator = separator
    
    def pos_tag_raw_text(self, text, as_tuple_list=True):
        # Unfortunately for the moment there is no method to do sentence split + pos tagging in nltk.parse.corenlp
        # Ony raw_tag_sents is available but assumes a list of str (so it assumes the sentence are already split)
        # We create a small custom function highly inspired from raw_tag_sents to do both

        parsed_text = self.parser(text)
        sentences = parsed_text.sentences

        def raw_tag_text():
            """
            Perform tokenizing sentence splitting and PosTagging and keep the 
            sentence splits structure
            """
            for tagged_sentence in sentences:
                yield [(token.text, token.xpos) 
                       for token in tagged_sentence.words]
        
        tagged_text = list(raw_tag_text())        

        if as_tuple_list:
            return tagged_text, sentences
        return '[ENDSENT]'.join(
            [' '.join([tuple2str(tagged_token, self.separator) for tagged_token in sent]) for sent in tagged_text]), sentences
        


