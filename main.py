"""
Generates questions from knowledge based or a corpus of text

sample usage:
-----------
python main.py -f data/pdpa/raw.csv
"""
import argparse
import pandas as pd

from src.kw_extractor import kw_extractor
from src.q_gen import q_gen
from src.utils import get_kw_sentence_pairs


# parse inputs
parser = argparse.ArgumentParser()
parser.add_argument("-f", "--filename", default='data/pdpa/raw.csv', help="name of file containing data")
# parser.add_argument("-s", "--savepath", default = 'outputs/output.csv', help="directory to save generated output")
args = parser.parse_args()

# read data as the last column of the dataframe
corpus = pd.read_csv(args.filename)
Series_of_texts = corpus.iloc[:,-1]
print(Series_of_texts.tail())
print(f"\nData read\n")


"""
1. Extract keywords
"""
# initiate models
kwe = kw_extractor()

# for each context, generate a dataframe and save it in generated_questions_list
list_of_qg_inputs_df = []
for text in Series_of_texts:

    # extract keywords
    kw, kw_rank, alias, sentences = kwe.extract_kw(text)

    # format to dataframe with and add sentence column
    context_w_kw = pd.DataFrame( kw, columns=['keywords'] ).assign(context = text)
    kw_sentence_pairs = get_kw_sentence_pairs(kw, sentences)
    kw_sentence_pairs = pd.DataFrame(kw_sentence_pairs, columns = ['keyword', 'sentence'])

    # add context column
    kw_sentence_context = kw_sentence_pairs.assign(context = text)

    # generate questions from keywords and context
    qg_inputs_df = kw_sentence_context.copy()
    context_qg_input = kw_sentence_context.context + ' [SEP] ' + kw_sentence_context.keyword
    qg_inputs_df = qg_inputs_df.assign(context_qg_input=context_qg_input)
    
    list_of_qg_inputs_df.append(qg_inputs_df)

aggregate_qg_inputs_df = pd.concat(list_of_qg_inputs_df).reset_index(drop=True)
print(f"\nKeywords extracted")
print(f"input dataframe shape: {aggregate_qg_inputs_df.shape}\n")


"""
2. Generate questions
"""
qg = q_gen()
gen_questions_kw_pair = qg.predict(aggregate_qg_inputs_df.context_qg_input)

gen_questions_str = [tup[0] for tup in gen_questions_kw_pair]
aggregate_qg_inputs_df = aggregate_qg_inputs_df.assign(gen_questions = gen_questions_str)
print(f"\Questions generated")
print(f"output dataframe shape: {aggregate_qg_inputs_df.shape}\n")


"""
3. Save output
"""
savefile_path = '/'.join( args.filename.split('/')[:-1] ) + '/output.csv'
aggregate_qg_inputs_df.to_csv(savefile_path)
print(f"\nOutputs saved in {savefile_path}")