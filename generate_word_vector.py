import jieba
import gensim
import pandas as pd
import numpy as np

class generate_doc_vector:
    '''
    all inputs must be a 1-dim np.array or list
    and return will be a dataframe ['company','vector']
    '''
    def __init__(self,company_list,doc_list,stopwords,keep_prob=1.0,drop_num=0):
        if len(company_list)!=len(doc_list):
            raise ValueError('Pls enter a matched document list')
        self.stopwords=list(stopwords)
        self.doc_list=[[word for word in jieba.cut(doc) if word not in self.stopwords] \
                            for doc in doc_list]
        self.lexicon=gensim.corpora.Dictionary(self.doc_list)##create dictionay in gensim
        self.lexicon.filter_extremes(no_below=drop_num,no_above=keep_prob)
        self.lexicon.compactify()
        self.bow = [self.lexicon.doc2bow(i) for i in self.doc_list]##transfer words to ind use doc2bow
        self.company_list=company_list
    
    def doc2vector(self,vector_size,alpha,window=5,min_alpha=0.0001,workers=3,\
        negative=10,dbow_words=0,min_count=5,dm=1):
        TaggedDocument=gensim.models.doc2vec.TaggedDocument
        corpus=[]
        for i, word in enumerate(self.doc_list):
            document=TaggedDocument(word,tags=[i])
            corpus.append(document)
        model=gensim.models.doc2vec.Doc2Vec(documents=corpus,dm=dm,dbow_words=dbow_words,\
                        vector_size=vector_size,epochs=6,window=window,alpha=alpha,\
                        min_alpha=min_alpha,min_count=min_count,negative=negative,workers=workers)
        model.train(corpus,total_examples=len(corpus),epochs=model.epochs)
        

        def _get_vectors(model,corpus):
            vec=[np.array(model.docvecs[i.tags[0]]).reshape((1,-1)) for i in corpus]
            return np.concatenate(vec)
        
        doc_vector=pd.DataFrame(_get_vectors(model,corpus))
        doc_vector.insert(0,'company',self.company_list)
        return doc_vector

    def bag_of_words(self,use_tfidf=True):
        if use_tfidf:
            tfidf=gensim.models.TfidfModel(self.bow)
            tfidf_bow=tfidf[self.bow]
            doc_vector=pd.DataFrame(gensim.matutils.corpus2dense(tfidf_bow,num_terms=len(self.lexicon.token2id)).T)
            doc_vector.insert(0,'company',self.company_list)
            return doc_vector
        else:
            doc_vector=pd.DataFrame(gensim.matutils.corpus2dense(self.bow,num_terms=len(self.lexicon.token2id)).T)
            doc_vector.insert(0,'company',self.company_list)
            return doc_vector

    def LSI(self,num_topics,use_tfidf=True):
        if use_tfidf:
            tfidf=gensim.models.TfidfModel(self.bow)
            tfidf_bow=tfidf[self.bow]
            lsi=gensim.models.LsiModel(corpus=tfidf_bow,num_topics=num_topics)
            doc_vector=pd.DataFrame(gensim.matutils.corpus2dense(lsi[tfidf_bow],num_terms=num_topics).T)
            doc_vector.insert(0,'company',self.company_list)
            return doc_vector
        else:
            lsi=gensim.models.LsiModel(corpus=self.bow,num_topics=num_topics)
            doc_vector=pd.DataFrame(gensim.matutils.corpus2dense(lsi[self.bow],num_terms=num_topics).T)
            doc_vector.insert(0,'company',self.company_list)
            return doc_vector

