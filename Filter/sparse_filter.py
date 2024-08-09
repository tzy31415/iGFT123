from rank_bm25 import BM25Okapi
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def TFIDFScore(query:str, document:str):
    tfidf_matrix = TfidfVectorizer().fit_transform([query, document])
    cosine_similarity = (tfidf_matrix[0] @ tfidf_matrix[1].T).toarray()[0][0]
    return cosine_similarity # The output range is [0,1], where 1 means very similar and 0 means very different


def BM25Score(query, gold_corpus, corpus):
    corpus = [gold_corpus] + corpus
    tokenized_corpus = [doc.split(" ") for doc in corpus]
    bm_corpus = BM25Okapi(tokenized_corpus)
    score = list(bm_corpus.get_scores(query.split(" ")))
    
    return score[0] if score.index(max(score)) == 0 else 0 # score if found, 0 if not

    
if __name__ == '__main__':
    corpus = [
    "Hello there good man!",
    "It is quite windy in London",
    "How is the weather today?"
    ]   
    query = "windy London"
    bm1 = TFIDFScore(query,"It is quite windy in London")
    print(bm1)



## 第一个的效果好一点