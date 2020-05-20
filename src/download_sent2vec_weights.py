import requests, zipfile, io

file_id_dict = {
    "sent2vec_wiki_unigrams":"0B6VhzidiLvjSa19uYWlLUEkzX3c",
    "sent2vec_wiki_bigrams ":"0B6VhzidiLvjSaER5YkJUdWdPWU0",
    "sent2vec_twitter_unigrams": "0B6VhzidiLvjSaVFLM0xJNk9DTzg",
    "sent2vec_twitter_bigrams": "0B6VhzidiLvjSeHI4cmdQdXpTRHc",
    "sent2vec_toronto books_unigrams":"0B6VhzidiLvjSOWdGM0tOX1lUNEk",
    "sent2vec_toronto books_bigrams":"0B6VhzidiLvjSdENLSEhrdWprQ0k"
}

# sent2vec_wiki_unigrams 
file_id = file_id_dict['sent2vec_wiki_unigrams']

s = requests.session()
r = s.get(f'https://docs.google.com/uc?export=download&id={file_id}')
confirm_code = r.text.split("/uc?export=download&amp;confirm=")[1].split("&amp;id=")[0]
r = s.get(f'https://docs.google.com/uc?export=download&confirm={confirm_code}&id={file_id}')
z = zipfile.ZipFile(io.BytesIO(r.content))
z.extractall()