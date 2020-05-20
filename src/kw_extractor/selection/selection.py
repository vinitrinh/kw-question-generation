"""
selection.py selects the keywords.
MMRPhrase is a the high level function.
_MMR is the key function in keyword selection.
"""
import warnings

from sklearn.metrics.pairwise import cosine_similarity

from ..embedding import *
from ..postagging import *


def embed_doc(embedding_distrib, text_obj, use_filtered=True):
    """
    Return the embedding of the full document

    Args:
    ----
        embedding_distrib: embedding distributor see @EmbeddingDistributor
        text_obj: input text representation see @InputTextObj
        use_filtered: if true keep only candidate words in the raw text before computing the embedding
    Return:
    -------
        document embedding: numpy array of shape (1, dimension of embeddings) 
    """
    if use_filtered:
        tagged = text_obj.filtered_pos_tagged
    else:
        tagged = text_obj.pos_tagged

    tokenized_doc_text = ' '.join(token[0].lower() for sent in tagged for token in sent)
    return embedding_distrib.get_tokenized_sents_embeddings([tokenized_doc_text])

def extract_candidates_embedding_for_doc(embedding_distrib, text_obj):
    """
    Return the list of candidate phrases as well as the associated numpy array that contains their embeddings.
    Note that candidates phrases extracted by PosTag rules  which are uknown (in term of embeddings)
    will be removed from the candidates.

    Args:
    -----
        embedding_distrib: embedding distributor see @EmbeddingDistributor
        text_obj: input text representation see @InputTextObj
    Return:
    -------
        candidate phrases: list of strings
        candidate phrases embeddings: a numpy array of shape (number of candidate phrases, dimension of embeddings :
                                      each row is the embedding of one candidate phrase
    """
    candidates = np.array(text_obj.extract_candidate_phrases())  # List of candidates based on PosTag rules
    if len(candidates) > 0:
        embeddings = np.array(embedding_distrib.get_tokenized_sents_embeddings(candidates))  # Associated embeddings
        valid_candidates_mask = ~np.all(embeddings == 0, axis=1)  # Only candidates which are not unknown.
        return candidates[valid_candidates_mask], embeddings[valid_candidates_mask, :]
    else:
        return np.array([]), np.array([])

def _MMR(embdistrib, text_obj, candidates, X, beta, N, use_filtered, alias_threshold):
    """
    Core method using Maximal Marginal Relevance in charge to return the top-N candidates.
    This contains the formula and the selection logic

    Args:
    ----

        embdistrib: embdistrib: embedding distributor see @EmbeddingDistributor
        text_obj: Input text representation see @InputTextObj
        candidates: list of candidates (string)
        X: numpy array with the embedding of each candidate in each row
        beta: hyperparameter beta for MMR (control tradeoff between informativeness and diversity)
        N: number of candidates to extract
        use_filtered: if true filter the text by keeping only candidate word before computing the doc embedding

    Return:
    -------
        N candidates: (list of string) list of the top-N candidates 
                      (or less if there are not enough candidates) 
        relevance_list: list of associated relevance scores (list of float)
        aliases_list: list containing for each keyphrase a list of alias 
                      (list of list of string)
    """

    # This is the key formula in the paper (2, 3a, 3b)
    N = min(N, len(candidates))
    doc_embedd = embed_doc(embdistrib, text_obj, use_filtered)  # Extract doc embedding
    doc_sim = cosine_similarity(X, doc_embedd.reshape(1, -1))

    doc_sim_norm = doc_sim/np.max(doc_sim)
    doc_sim_norm = 0.5 + (doc_sim_norm - np.average(doc_sim_norm)) / np.std(doc_sim_norm)

    sim_between = cosine_similarity(X)
    np.fill_diagonal(sim_between, np.NaN)

    sim_between_norm = sim_between/np.nanmax(sim_between, axis=0)
    sim_between_norm = \
        0.5 + (sim_between_norm - np.nanmean(sim_between_norm, axis=0)) / np.nanstd(sim_between_norm, axis=0)

    # Loop through the candidates and pick the top choice till we reach N keyphrases
    # initialize a list to take the first keyphrase
    # this is part of the larger loop below
    selected_candidates = []
    unselected_candidates = [c for c in range(len(candidates))]
    j = np.argmax(doc_sim)
    selected_candidates.append(j)
    unselected_candidates.remove(j)

    for _ in range(N - 1):
        selec_array = np.array(selected_candidates)
        unselec_array = np.array(unselected_candidates)

        distance_to_doc = doc_sim_norm[unselec_array, :]
        dist_between = sim_between_norm[unselec_array][:, selec_array]
        if dist_between.ndim == 1:
            dist_between = dist_between[:, np.newaxis]
        j = np.argmax(beta * distance_to_doc - (1 - beta) * np.max(dist_between, axis=1).reshape(-1, 1))
        item_idx = unselected_candidates[j]
        selected_candidates.append(item_idx)
        unselected_candidates.remove(item_idx)

    # Not using normalized version of doc_sim for computing relevance
    relevance_list = max_normalization(doc_sim[selected_candidates]).tolist()
    aliases_list = get_aliases(sim_between[selected_candidates, :], candidates, alias_threshold)

    return candidates[selected_candidates].tolist(), relevance_list, aliases_list

def MMRPhrase(embdistrib, text_obj, beta=0.65, N=10, use_filtered=True, alias_threshold=0.8):
    """
    Extract N keyphrases

    Args:
    ----
        embdistrib: embedding distributor see @EmbeddingDistributor
        text_obj: Input text representation see @InputTextObj
        beta: hyperparameter beta for MMR (control tradeoff between informativeness and diversity)
        N: number of keyphrases to extract
        use_filtered: if true filter the text by keeping only candidate word before computing the doc embedding
    
    Return:
    -------
        N candidates: (list of string) list of the top-N candidates 
                      (or less if there are not enough candidates) 
        relevance_list: list of associated relevance scores (list of float)
        aliases_list: list containing for each keyphrase a list of alias 
                      (list of list of string)
    """
    candidates, X = extract_candidates_embedding_for_doc(embdistrib, text_obj)

    if len(candidates) == 0:
        warnings.warn('No keyphrase extracted for this document')
        return None, None, None

    return _MMR(embdistrib, text_obj, candidates, X, beta, N, use_filtered, alias_threshold)


def max_normalization(array):
    """
    Compute maximum normalization (max is set to 1) of the array
    Args:
    -----
        array: 1-d array
    Return:
    ------
        1-d array max- normalized : each value is multiplied by 1/max value
    """
    return 1/np.max(array) * array.squeeze(axis=1)


def get_aliases(kp_sim_between, candidates, threshold):
    """
    Find candidates which are very similar to the keyphrases (aliases).
    The threshold is based on cosine similarity. 
    ie, threshold of 0.8 identifies alias 
        that have a cosine similarity of 0.8 to the selected phrases
        which is quite a high similarity
    
    Args:
    -----
        kp_sim_between: ndarray of shape (nb_kp , nb candidates) containing the similarity
                        of each kp with all the candidates. 
                        Note that the similarity between the keyphrase and itself should be set to
                        NaN or 0
        candidates: array of candidates (array of string)
    
    Return:
    ------
        list containing for each keyphrase a list that contain candidates which are aliases
        (very similar) (list of list of string)
    """

    kp_sim_between = np.nan_to_num(kp_sim_between, 0)
    idx_sorted = np.flip(np.argsort(kp_sim_between), 1)
    aliases = []
    for kp_idx, item in enumerate(idx_sorted):
        alias_for_item = []
        for i in item:
            if kp_sim_between[kp_idx, i] >= threshold:
                alias_for_item.append(candidates[i])
            else:
                break
        aliases.append(alias_for_item)

    return aliases
