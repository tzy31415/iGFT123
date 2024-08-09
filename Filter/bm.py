from filter.sparse_filter import BM25Score
from tqdm import tqdm
import json
import random
import torch




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



def write_json(data, json_file):
    with open(json_file, 'w') as f:
        json.dump(data, f, indent=4)


def get_score1(dataset, bm_num=200):
    pseudo_query = f'F:\\Dataset\\pseudo\\{dataset}_generate_sft_per1.json'
    corpus_file = f"F:\\Dataset\\{dataset}\\corpus.jsonl"
    
    corpus_data = read_jsonl(corpus_file)
    query_data = get_coprpus_all_files(pseudo_query)
    
    result = []
    gold_id_set = set([query['cid'] for query in query_data])
    # selected_texts = random.sample(list({cid: text for cid, text in corpus_data.items() if cid not in  gold_id_set}.values()), bm_num)
    selected_texts = random.sample(list({cid: text for cid, text in corpus_data.items()}.values()), bm_num)

    for query_item in tqdm(query_data):
        generated_query = query_item['pseudo_query']
        gold_id = query_item['cid']
        gold_corpus = query_item['corpus']
        
        score = BM25Score(generated_query,gold_corpus,selected_texts)
        result.append({
           'score':score,
           'query':generated_query,
           'corpus':gold_corpus,
           'cid':gold_id 
        })
        
    write_json(result,f'F:\\Dataset\\bm\\{dataset}_generate_sft_per1.json')
    
    
    
    
def get_score(dataset):
    bm_query = f'F:\\Dataset\\bm\\{dataset}_generate_sft_per1.json'
    query_data = get_coprpus_all_files(bm_query)
    count = 0
    for i in query_data:
        if i['score'] == 0:
            count += 1
    print(f'{dataset}:{count/len(query_data)}, total:{len(query_data)-count}')




if __name__ == "__main__":
    # # get_score('arguana')
    # get_score('climate-fever')
    # get_score('dbpedia-entity')
    # get_score('fever')
    # get_score('hotpotqa')
    # get_score('nfcorpus')
    # get_score('quora')
    # get_score('scidocs')
    # get_score1('scifact')
    get_score1('trec-covid')
    get_score1('webis-touche2020')

    get_score('trec-covid')
    get_score('webis-touche2020')