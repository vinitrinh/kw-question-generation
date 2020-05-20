def get_kw_sentence_pairs(kw, sentences):
    """
    Pair the keywords and their sentences
    Allows for multiple sentences
    
    args:
    ----
        keyword: (list of str)
        sentence: (list of str)
    return:
    ------
        list of keyword-sentence tuple pairs 
    """
    return [(keyword, sentence) 
            for keyword in kw 
            for sentence in sentences 
            if keyword.lower() in sentence.lower()]