from filter.sparse_filter import BM25Score
from tqdm import tqdm
import json
import random
import csv
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, T5ForConditionalGeneration, T5Tokenizer
import torch
import numpy as np


class MonoT5():
    def __init__(self,model_name = '/gemini/code/monoT5') -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(self.device)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        
    def retrieve(self, query, doc):
        input_text = f"Query: {query} Document: {doc}"
        
        input_text = self.tokenizer.encode(input_text, return_tensors='pt').to(self.device)
    
        with torch.no_grad():
            outputs = self.model.generate(input_text, return_dict_in_generate=True, output_scores=True,output_logits=True)
        decoded_output = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True) # 输出的true或者false
        
        prob_list = outputs.logits[0].cpu().numpy().flatten().tolist()
        true_value = prob_list[1176]
        false_value = prob_list[6136]
    
        if decoded_output == 'false':
            return  0
        else:
            return float(np.exp(true_value)/(np.exp(false_value)+np.exp(true_value)))
     



def read_jsonl(filename):
    id2doc = {}
    with open(filename, 'r') as f:
        for line in tqdm(f):
            item = json.loads(line)
            id1 = item['_id']
            text = item['text']
            id2doc[id1] = text
    return id2doc

        
def get_coprpus_all_files(filename):
    result = []
    with open(filename, 'r') as f:
        result = json.load(f)
    return result



def read_tsv(filename):
    with open(filename, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter='\t')
        qid2cid = []
        for row in tqdm(reader):
            if row[2] == 'score':
                continue
            qid2cid.append([row[0],row[1]])
    return qid2cid
            
def write_json(data, json_file):
    with open(json_file, 'w') as f:
        json.dump(data, f, indent=4)



def write_tsv_line_by_line(data, file):
    with open(file, 'w',encoding='utf-8') as file:
        writer = csv.writer(file, delimiter='\t')
        # writer.writerow(['query-id', 'corpus-id', 'score'])
        for qid,cid,rank in data:
            writer.writerow([qid, cid, rank])
    

        
def process_qrel(data):
    qid2cid_rank = {}
    
    for item in data:
        qid, cid = item
        if qid not in qid2cid_rank.keys():
            qid2cid_rank[qid] = [cid]
        else:
            qid2cid_rank[qid].append(cid)
    return qid2cid_rank
        
def tsv_reader(input_filepath):
    reader = csv.reader(open(input_filepath, encoding="utf-8"), delimiter="\t", quoting=csv.QUOTE_MINIMAL)
    for idx, row in enumerate(reader):
        yield idx, row


def lineid2cid(corpus):
    id2doc = {}
    with open(corpus, 'r') as f:
        for idx, line in tqdm(enumerate(f)):
            item = json.loads(line)
            id1 = item['_id']
            text = item['text']
            id2doc[str(idx)] = id1
    return id2doc


def mymono(dataset):
    print("*"*100)
    print(f'now dataset is {dataset}')
    print("*"*100)
    # 应该都是dev数据集
    corpus_file = f"E:\\Dataset\\{dataset}\\corpus.jsonl"
    query_file = f"E:\\Dataset\\{dataset}\\queries.jsonl"
    qrels_file = f"E:\\Dataset\\{dataset}\\qrels\\dev.tsv"
    ranking_file = f'E:\\ranking\\{dataset}_ranking.tsv'
        
    corpus_data = read_jsonl(corpus_file)
    query_data = read_jsonl(query_file)
    test_data = read_tsv(qrels_file)
    rank_data = read_tsv(ranking_file)
    lid2cid = lineid2cid(corpus_file)
    
    result = []
    rank_data = process_qrel(rank_data)
    
    test_data = process_qrel(test_data)
    count = 0
    
    
    
    for qid in tqdm(rank_data.keys()):
        cid_list = rank_data[qid]
        query = query_data[qid]
        temp = {}
        score = 10
        grouth_truth_cid = test_data[qid]
        for cid in cid_list:
            cid1 = lid2cid[cid]
            if cid1 in grouth_truth_cid:
                temp[cid] = 100
                count += 1
            else:
                temp[cid] = score
                score -=1 
                
                
            # cid1 = lid2cid[cid]
            # corpus = corpus_data[cid1]
            # score = mono.retrieve(query,corpus)
            # temp[cid] = score
        temp = dict(sorted(temp.items(), key=lambda item: item[1], reverse=True))
        rank = 1
        for cid in temp:
            result.append([qid,cid,rank])
            rank += 1
    print(f'count:{count}')
    write_tsv_line_by_line(result,f'E:\\ranking_mono\\{dataset}_my.tsv')
            

        
            
            
            
    
def mono(dataset, mono):
    print("*"*100)
    print(f'now dataset is {dataset}')
    print("*"*100)
    # 应该都是dev数据集
    corpus_file = f"E:\\Dataset\\{dataset}\\corpus.jsonl"
    query_file = f"E:\\Dataset\\{dataset}\\queries.jsonl"
    qrels_file = f"E:\\Dataset\\{dataset}\\qrels\\dev.tsv"
    ranking_file = f'E:\\ranking\\{dataset}_ranking.tsv'
        
    
    corpus_data = read_jsonl(corpus_file)
    query_data = read_jsonl(query_file)
    test_data = read_tsv(qrels_file)
    rank_data = read_tsv(ranking_file)
    lid2cid = lineid2cid(corpus_file)
    
    
    result = []
    rank_data = process_qrel(rank_data)
    for qid in tqdm(rank_data.keys()):
        cid_list = rank_data[qid]
        query = query_data[qid]
        temp = {}
        for cid in cid_list:
            cid1 = lid2cid[cid]
            
            corpus = corpus_data[cid1]
            score = mono.retrieve(query,corpus)
            temp[cid] = score
        temp = dict(sorted(temp.items(), key=lambda item: item[1], reverse=True))
        rank = 1
        for cid in temp:
            result.append([qid,cid,rank])
            rank += 1
    write_tsv_line_by_line(result,f'E:\\ranking_mono\\{dataset}_ranking_{mono_type}.tsv')
            

        
            
    
    
    
    

    
    
    
    
def get_score(dataset):
    bm_query = f'F:\\Dataset\\bm\\{dataset}_generate_sft_per1.json'
    query_data = get_coprpus_all_files(bm_query)
    count = 0
    for i in query_data:
        if i['score'] == 0:
            count += 1
    print(f'{dataset}:{count/len(query_data)}, total:{len(query_data)-count}')




if __name__ == "__main__":
    # mono('arguana')
    # mono('climate-fever')
    # mono('dbpedia-entity')

    # # # mono_type = 'base'
    # # # mono_path = f'E:\\monoT5-{mono_type}'
    # # # mono_model = MonoT5(mono_path)

    # mono('hotpotqa')
    # mono('nfcorpus')
    # mono('quora')
    # mono('scidocs')
    # mono('scifact')
    # mono('fiqa')
    # mono('trec')
    # mono('touch')
    # mono('msmarco')
    # mono('fever')



    # mono_type = 'large'
    # mono_path = f'E:\\monoT5-{mono_type}'
    # mono_model = MonoT5(mono_path)
    # mono('hotpotqa',mono_model)
    # mono('nfcorpus',mono_model)
    # mono('quora',mono_model)
    # mono('scidocs',mono_model)
    # mono('scifact',mono_model)
    # mono('fiqa',mono_model)
    # mono('trec',mono_model)
    # mono('touch',mono_model)
    # mono('msmarco',mono_model)
    # # mono('fever',mono_model)
    # mono('dbpedia-entity',mono_model)
    # mono('arguana',mono_model)
    # mono('climate-fever',mono_model)
    
    
    
    mono_type = '3B'
    mono_path = f'E:\\monoT5-{mono_type}'
    mono_model = MonoT5(mono_path)

    # mono('hotpotqa',mono_model)
    # mono('nfcorpus',mono_model)
    mono('quora',mono_model)
    mono('scidocs',mono_model)
    mono('scifact',mono_model)
    mono('fiqa',mono_model)
    mono('trec',mono_model)
    mono('touch',mono_model)
    mono('msmarco',mono_model)
    # mono('fever',mono_model)
    mono('dbpedia-entity',mono_model)
    mono('arguana',mono_model)
    mono('climate-fever',mono_model)