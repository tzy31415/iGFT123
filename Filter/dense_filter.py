from time import time
from beir import util, LoggingHandler
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import numpy as np
# pip install pytrec_eval beir==1.0.1

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, T5ForConditionalGeneration, T5Tokenizer
import torch
import numpy as np
from torch.nn.functional import softmax


class DRPScore():
    def __init__(self,model_save_path = '/gemini/code/bert') -> None:
        self.model = DRES(models.SentenceBERT(model_save_path), batch_size=256, corpus_chunk_size=100000)
        self.retriever = EvaluateRetrieval(self.model, k_values=[10], score_function="cos_sim")
        
    def retrieve(self, query, gold_corpus, corpus):
        query_id = "qid"
        query_list = {query_id:query, 'a':'b'}
        corpus = [gold_corpus] + corpus
        corpus = {str(i): {"text": doc} for i, doc in enumerate(corpus)}
        
        result = self.retriever.retrieve(corpus, query_list)
        score = list(result[query_id].values())
        if score.index(max(score)) == 0:
            return score[0]
        else:
            return 0

class USBert(): # 这个模型对于bert类的模型都可以进行操作，或者colbert也可以这么操作
    # https://huggingface.co/sentence-transformers/use-cmlm-multilingual
    def __init__(self, model_save_path = '/gemini/code/roberta'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_save_path)
        self.model = AutoModel.from_pretrained(model_save_path).to(self.device)

    def embed(self, sentences):
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt').to(self.device)
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = encoded_input['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
        sentence_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sentence_embeddings

    def retrieve(self, query, gold_corpus, doc):
        doc = [gold_corpus] + doc
        query_embeddings = self.embed(query)
        doc_embeddings = self.embed(doc)
        query_embeddings = query_embeddings.expand_as(doc_embeddings)
        similarities_list = F.cosine_similarity(query_embeddings, doc_embeddings, dim=1).tolist()
        
        
        return  similarities_list[0] if similarities_list.index(max(similarities_list)) == 0 else 0


class MonoT5():
    def __init__(self,model_name = '/gemini/code/monoT5') -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(self.device)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        
    def retrieve(self, query, doc,doc1):
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
     



# 输出的是 true或者false



# class RoBERTa():
#     # https://huggingface.co/sentence-transformers/stsb-roberta-base
#     def __init__(self,model_save_path) -> None:
        

# if __name__ == '__main__':
#     model = USBert('/gemini/code/use')
#     query = 'Where the capital of Spain'
#     corpus = [
#         "Berlin is the capital of Germany.",
#         "Madrid is the capital of Spain.",
#         "Rome is the capital of Italy.",
#         "London is the capital of the United Kingdom."
#     ]
    
#     score = model.score(query,corpus)
#     print(score)

    




# if __name__ == "__main__":
#     model_path = '/gemini/code/bert'
#     dpr = DRPScore(model_path)
#     query = 'What is the capital of France?'
#     gold = "Paris is the capital city of France."
#     # Example corpus (list of documents)
#     corpus = [
#         "Berlin is the capital of Germany.",
#         "Madrid is the capital of Spain.",
#         "Rome is the capital of Italy.",
#         "London is the capital of the United Kingdom."
#     ]

#     result = dpr.retrieve(query,gold,corpus)
#     print(result)