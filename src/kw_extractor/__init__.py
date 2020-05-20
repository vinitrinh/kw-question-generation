"""
Unsupervised Keyword extractor:
    https://arxiv.org/abs/1801.04470
"""
from sklearn.metrics.pairwise import cosine_similarity

from .embedding import use_enc, sent2vec_enc
from .postagging import InputTextObj, PosTaggingStanza
from .selection import MMRPhrase, max_normalization, get_aliases
from .config import CONFIG


class kw_extractor:
    """
    Sample usage:
    ------------
    >>> from src.kw_extractor import kw_extractor
    >>> text = "The Normans thereafter adopted the growing feudal doctrines of the rest of France, andworked them into a functional hierarchical system in both Normandy and in England.  The new Normanrulers were culturally and ethnically distinct from the old French aristocracy, most of whom traced theirlineage to Franks of the Carolingian dynasty. Most Norman knights remained poor and land-hungry, and by1066 Normandy had been exporting fighting horsemen for more than a generation. Many Normans of Italy,France and England eventually served as avid Crusaders under the Italo-Norman prince Bohemund I and the Anglo-Norman king Richard the Lion-Heart."
    >>> kwe = kw_extractor()
    >>> kwe.extract_kw(text, beta=0.9)

    (['normans',
    'functional hierarchical system',
    'avid crusaders',
    'franks',
    'france',
    'rest',
    'norman prince bohemund',
    'new normanrulers',
    'lion',
    'normandy'],
    [1.0,
    0.2651855945587158,
    0.30983009934425354,
    0.21946312487125397,
    0.3710859715938568,
    0.0028201609384268522,
    0.8210158348083496,
    -0.1238916888833046,
    0.13850507140159607,
    0.5787600874900818],
    [[], [], [], [], [], [], [], [], [], []])
    """
    def __init__(self):
        self.__dict__.update(CONFIG)
        self.postagger = PosTaggingStanza()

        if self.model == "use":
            self.encoder = use_enc()
        elif self.model == "sent2vec":
            self.encoder = sent2vec_enc(self.fasttext_model)
        else:
            raise ValueError("model not recognized")

        

    def extract_kw(self, text, alias_threshold=1):
        """
        args:
        ----
            text: (str) clause or context to extract keywords from

        Return:
        ------
            kw: (list of str)
            kw_rank: (iterable of float)
            alias: (list of list of str)
            sentences: (list of str)
        """
        tagged, sentences = self.postagger.pos_tag_raw_text(text)
        sentences = [sentence.text for sentence in sentences]
        text_obj = InputTextObj(tagged)

        kw, kw_rank, alias = MMRPhrase(self.encoder, text_obj, 
                                       N=self.N, beta=self.beta, 
                                       alias_threshold=self.alias_threshold)
        return kw, kw_rank, alias, sentences
