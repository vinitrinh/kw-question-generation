# KWIG! A Quick Keyword Based Question Generator that is fast and convenient


__Why the hell would anyone need to generate questions?__  
Seomtimes, people do it to train their QA questions. In order to train QA systems, engineers need not just a large corpus but human generated questions matched to their human labelled answer spans. This is extremely tedious. It has been found that generated questions can really improve QA systems (see UniLM paper below). 
</br>  
This repo generates questions from a corpus or knowledge base using keywords. Importantly, no training is needed as it used pre-trained models.  

__Do QA systems really benefit from synthetic QA data?__  
Yes! Finetuning [AI Singapore's Golden Retriever](https://github.com/aimakerspace/goldenretriever) on questions from the PDPA dataset typically worsens OOS performance. This is because the pathetic dataset is at most 200 question answer pairs; it does not matter how you split the data, finetuning on that small a dataset will worsen its generalizability. (You can try it with the link) However, when fintuning on the numerous 3000 synthetic questions, the OOS performance improves. Very nicely below, we enjoy a significant improve in recall@2 score.   

<img src="img/0.1 margin finetune on synthetic.png">

## Installation  
Create an environment, perhaps with Conda:  
`conda create --name qgen`  
`conda activate qgen`  
</br>
Install using `setup.sh` which will install `requirements.txt` and Sent2Vec dependencies.    
`bash setup.sh`  

## Usage  
To run the main script, run from the following:  
`python main.py -f data/pdpa/raw.csv`  

## Model
Keywords are extracted from contexts (paragraphs) using Google's Universal Sentence Encoder (USE).  
Keywords are inputs to the question generator.  
<img src="img/Keyword Based Question Generator.jpeg">

## Examples
Input Context: __Personal data refers to data, whether true or not, about an individual who can be identified from that data; or from that data and other information to which the organisation has or is likely to have access.__ This includes unique identifiers (e.g. NRIC number, passport number)...
</br>
Generated Query: What is personal data?  
</br>
Input Context: ...other information to which the organisation has or is likely to have access. __This includes unique identifiers (e.g. NRIC number, passport number); photographs or video images of an individual (e.g. CCTV images); as well as any set of data (e.g. name, age, address, telephone number, occupation, etc), which when taken together would be able to identify the individual.__ For example, Jack Lim, 36 years old, civil servant, lives at Blk 123 Bishan St 23.
</br>
Generated Query: What is included in personal data?  
</br>
Input Context: ...The data protection provisions govern the collection, use and disclosure of personal data by organisations. __In brief, the PDPA contains three main sets of data protection obligations:__ Obligations relating to notification, consent and purpose. ...
</br>
Generated Query: What does the PDPA contain?  

## References and Acknowledgements
Bennani-Smires, Kamil, et al. "Simple unsupervised keyphrase extraction using sentence embeddings." arXiv preprint arXiv:1801.04470 (2018).  
https://github.com/swisscom/ai-research-keyphrase-extraction  
</br>
Dong, Li, et al. "Unified language model pre-training for natural language understanding and generation." Advances in Neural Information Processing Systems. 2019.  
https://github.com/microsoft/unilm  
</br>
Cer, Daniel, et al. "Universal sentence encoder." arXiv preprint arXiv:1803.11175 (2018).  
</br>
Gupta, Prakhar, Matteo Pagliardini, and Martin Jaggi. "Better word embeddings by disentangling contextual n-gram information." arXiv preprint arXiv:1904.05033 (2019).  
