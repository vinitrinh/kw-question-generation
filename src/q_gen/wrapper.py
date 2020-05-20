import glob
import math
from tqdm import tqdm
import numpy as np
import torch
import random
import requests, zipfile, io
import os

from .pytorch_pretrained_bert.tokenization import BertTokenizer, WhitespaceTokenizer
from .pytorch_pretrained_bert.modeling import BertForSeq2SeqDecoder
from .config import CONFIG, STOP_WORDS

from .biunilm import seq2seq_loader

def detokenize(tk_list):
    r_list = []
    for tk in tk_list:
        if tk.startswith('##') and len(r_list) > 0:
            r_list[-1] = r_list[-1] + tk[2:]
        else:
            r_list.append(tk)
    return r_list

class q_gen(object):
    """
    Sample usage:
    -----------
    >>> from src.q_gen import q_gen
    >>> qg = q_gen()
    >>> qg.predict(["He is an alien", "Where is the spaceship?"])
    """
    def __init__(self, **kwargs):
        """
        Inherits all params from config.py

        Major initialization processes include:
            - bert tokenizer
            - preprocesser in self.bi_uni_pipeline
            - download_pretrained_model
            - init model object BertForSeq2SeqDecoder
        """
        # inherit config params
        self.__dict__.update(CONFIG)
        self.__dict__.update(kwargs)

        if self.max_tgt_length >= self.max_seq_length - 2:
            raise ValueError("Maximum tgt length exceeds max seq length - 2.")

        self.device = torch.device("cpu")
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count()

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if n_gpu > 0:
            torch.cuda.manual_seed_all(self.seed)

        self.tokenizer = BertTokenizer.from_pretrained("bert-large-cased", do_lower_case=self.do_lower_case)

        self.tokenizer.max_len = self.max_seq_length

        pair_num_relation = 0
        self.bi_uni_pipeline = []
        self.bi_uni_pipeline.append(seq2seq_loader.Preprocess4Seq2seqDecoder(list(self.tokenizer.vocab.keys()), 
                                                                             self.tokenizer.convert_tokens_to_ids, 
                                                                             self.max_seq_length, 
                                                                             max_tgt_length=self.max_tgt_length, 
                                                                             new_segment_ids=self.new_segment_ids,
                                                                             mode="s2s", 
                                                                             num_qkv=0, 
                                                                             s2s_special_token=self.s2s_special_token, 
                                                                             s2s_add_segment=self.s2s_add_segment, 
                                                                             s2s_share_segment=self.s2s_share_segment, 
                                                                             pos_shift=self.pos_shift))

        # Prepare model
        cls_num_labels = 2
        type_vocab_size = 6 + (1 if self.s2s_add_segment else 0) if self.new_segment_ids else 2
        mask_word_id, eos_word_ids, sos_word_id = self.tokenizer.convert_tokens_to_ids(["[MASK]", "[SEP]", "[S2S_SOS]"])

        self.download_pretrained_model()

        # LOAD THE ENCODER & DECODER
        print("***** Recover model: %s *****", self.model_recover_path)
        
        map_device='cpu'
        model_recover = torch.load(self.model_recover_path,map_location=map_device)
        self.model = BertForSeq2SeqDecoder.from_pretrained("bert-large-cased", 
                                                            state_dict=model_recover, 
                                                            num_labels=cls_num_labels, 
                                                            num_rel=pair_num_relation, 
                                                            type_vocab_size=type_vocab_size, 
                                                            task_idx=3, 
                                                            mask_word_id=mask_word_id, 
                                                            search_beam_size=self.beam_size,
                                                            length_penalty=self.length_penalty,
                                                            eos_id=eos_word_ids, 
                                                            sos_id=sos_word_id, 
                                                            forbid_duplicate_ngrams=self.forbid_duplicate_ngrams, 
                                                            forbid_ignore_set=None, 
                                                            not_predict_set=None, 
                                                            ngram_size=self.ngram_size, 
                                                            min_len=self.min_len, 
                                                            mode=self.mode,
                                                            max_position_embeddings=self.max_seq_length, 
                                                            ffn_type=0, num_qkv=0,
                                                            seg_emb=False, pos_shift=self.pos_shift)
        del model_recover

        self.model.to(self.device)

        # if n_gpu > 1:
        #     self.model = torch.nn.DataParallel(self.model)
        # torch.cuda.empty_cache()

        self.model.eval()


    def download_pretrained_model(self):
        """
        The logic of the model is to check 
        if the weights are already saved offline, 
        then downlaod the weights
        """
        if os.path.isfile(self.model_recover_path):
            # Model is already saved
            print(f"{self.model_recover_path} found in current directory.")
            return
        else:
            # Download model
            s = requests.session()
            file_id = self.file_id
            r = s.get(f'https://docs.google.com/uc?export=download&id={file_id}')
            confirm_code = r.text.split("/uc?export=download&amp;confirm=")[1].split("&amp;id=")[0]
            r = s.get(f'https://docs.google.com/uc?export=download&confirm={confirm_code}&id={file_id}')
            z = zipfile.ZipFile(io.BytesIO(r.content))
            z.extractall()
            return

    def _get_answer_tokens(self, tkns):
        """
        this function gets a random answer from a string sentence
        separated by a 'SEP' token 

            Sample input: ['I', 'went', 'to', 'school','.']
            Sample output: ['I', 'went', 'to', 'school','.'] + ['SEP'] + ['school'] 
        """
        words = detokenize(tkns)
        answers = []
        for w in words:
            if len(w) > 1:
                if w.lower() not in STOP_WORDS:
                    answers.append(w)
        return self.tokenizer.tokenize(random.choice(answers) if answers else words[0])

    
    def predict(self, input_lines):
        """
        Generate questions from clauses

        qg.predict(["I will go to school today to take my math exam. [SEP] exam"])
        >>> [('What will I take after school?', 'exam')]
        """
        data_tokenizer = self.tokenizer
        max_src_length = self.max_seq_length - 2 - self.max_tgt_length
        input_lines = [data_tokenizer.tokenize(x)[:max_src_length] for x in input_lines]

        # ensure an answer in each string of input_lines
        input_lines = [x + ["[SEP]"] + self._get_answer_tokens(x) if "[SEP]" not in x else x for x in input_lines]

        # input lines are sorted by their length
        # but the indices provided by enumerate will reorder the output lines correctly later
        input_lines = sorted(list(enumerate(input_lines)), key=lambda x: -len(x[1]))
        output_lines = [""] * len(input_lines)
        total_batch = math.ceil(len(input_lines) / self.batch_size)

        next_i = 0
        with tqdm(total=total_batch) as pbar:
            while next_i < len(input_lines):

                # essentially _chunk is one batch of data
                _chunk = input_lines[next_i:next_i + self.batch_size]

                # buf and buf_id are the inputs indexed accordingly
                buf_id = [x[0] for x in _chunk]
                buf = [x[1] for x in _chunk]

                # update batch tracker
                next_i += self.batch_size

                # bi_uni_pipeline contains unilim's seq2seq-preprocessor
                # max_a_len helps in the looped logic
                # the seq2seq preprocessor returns 6 preprocessed inputs
                # input_ids, segment_ids, position_ids, input_mask, mask_qkv, self.task_idx
                # input_mask is a mystery, its a 315x315 tensor
                # mask_qkv is None
                max_a_len = max([len(x) for x in buf])
                instances = []
                for instance in [(x, max_a_len) for x in buf]:
                    for proc in self.bi_uni_pipeline:
                        instances.append(proc(instance))

                with torch.no_grad():
                    batch = seq2seq_loader.batch_list_to_batch_tensors(instances)
                    batch = [t.to(self.device) if t is not None else None for t in batch]
                    input_ids, token_type_ids, position_ids, input_mask, mask_qkv, task_idx = batch
                    traces = self.model(input_ids, token_type_ids,
                                    position_ids, input_mask, task_idx=task_idx, mask_qkv=mask_qkv)

                    if self.beam_size > 1:
                        traces = {k: v.tolist() for k, v in traces.items()}
                        output_ids = traces['pred_seq']
                    else:
                        output_ids = traces.tolist()

                    for i in range(len(buf)):
                        w_ids = output_ids[i]
                        output_buf = self.tokenizer.convert_ids_to_tokens(w_ids)
                        output_tokens = []

                        for t in output_buf:
                            if t in ("[SEP]", "[PAD]"):
                                break
                            output_tokens.append(t)
                        
                        output_sequence = ' '.join(detokenize(output_tokens))
                        output_sequence = output_sequence.replace(" ' ", "'").replace(" ?", "?")
                            
                        ans_idx = buf[i].index("[SEP]")
                        corresponding_answer = ' '.join(detokenize(buf[i][ans_idx+1:]))
                        output_lines[buf_id[i]] = (output_sequence, corresponding_answer)             

                pbar.update(1)

        return output_lines