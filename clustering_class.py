import pandas as pd
import numpy as np
import functools
import jieba
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture

class bag_of_words:
    ##require pandas DataFrame
    def __init__(self, company_list, company_disc):
        assert len(company_list) == company_disc.shape[0]
        self.company_list = company_list.copy()
        self.company_disc = company_disc.copy()
        self.word_embedding = None

    def transform(self, stopwords, use_tfidf=True):
        words = self.company_disc.apply(lambda x: " ".join(jieba.cut(x)))
        cv = CountVectorizer(stop_words=stopwords)
        word_embedding = cv.fit_transform(words)
        if use_tfidf:
            tf = TfidfTransformer()
            word_embedding = tf.fit_transform(word_embedding)
        self.word_embedding = pd.concat(
            [self.company_list, pd.DataFrame(word_embedding.toarray())], axis=1
        )

    def get_vector(self):
        return self.word_embedding


class paper_cluster:
    def __init__(self, company_list, company_vec):
        assert len(company_list) == company_vec.shape[0]
        self.company_list = company_list.copy()
        self.cosine_matrix = cosine_similarity(company_vec)
        self.cluster_dict = dict()
        self.name = "paper_class"
        self.paper_class = None

    def generate_clusters(self, n_clusters=10, adjusted_coef=1):

        cov_matrix_enlarge = self.cosine_matrix.copy()
        max_num = self.cosine_matrix.shape[0]
        enlarge_company_list = self.company_list.copy()

        # query all company in the same industry
        def query_ind(index):
            if type(enlarge_company_list[index]) == type("a"):
                yield index
            else:
                sub_index = enlarge_company_list[index]
                for i in sub_index:
                    for j in query_ind(i):
                        yield j

        # sum correlation
        def sum_corr_func(index1, index2):
            return cov_matrix_enlarge[index1, index2]

        # set the corr of companies within the same industry to 1
        def elim_corr(comp_list):
            for i in comp_list:
                cov_matrix_enlarge[i, :] = 0
                cov_matrix_enlarge[:, i] = 0

        set_num_counter = list(range(max_num))
        exempt_set = set()
        record_dummy = -1
        for i in range(max_num):
            cov_matrix_enlarge[i, i] = 0

        while len(set(set_num_counter)) > n_clusters:
            # get minimun correlation in the matrix
            min_pair = cov_matrix_enlarge.argmax()
            row = min_pair // cov_matrix_enlarge.shape[0]
            col = min_pair % cov_matrix_enlarge.shape[0]
            enlarge_company_list.append([row, col])
            # calculate the new generated industry correlation with other companies
            ind_comp_list = [i for i in query_ind(row)] + [i for i in query_ind(col)]
            for i in ind_comp_list:
                exempt_set.add(i)
            exempt_set.add(row)
            exempt_set.add(col)
            new_corr = []
            for iter_company in range(cov_matrix_enlarge.shape[1]):
                if_exempt = iter_company in exempt_set
                if (iter_company >= max_num) and (not if_exempt):
                    iter_comp_com = [i for i in query_ind(iter_company)]
                    sum_corr = (
                        sum(
                            [
                                self.cosine_matrix[i, j]
                                for i in iter_comp_com
                                for j in ind_comp_list
                            ]
                        )
                        / (len(iter_comp_com) * len(ind_comp_list)) ** adjusted_coef
                    )
                elif not if_exempt:
                    sum_corr = sum(
                        [sum_corr_func(i1, iter_company) for i1 in ind_comp_list]
                    ) / len(ind_comp_list)
                else:
                    sum_corr = 0
                new_corr.append(sum_corr)
            cov_matrix_enlarge = np.vstack((cov_matrix_enlarge, np.matrix(new_corr)))
            cov_matrix_enlarge = np.hstack(
                (cov_matrix_enlarge, np.transpose(np.matrix(new_corr + [0])))
            )
            # set the corr within the same industry to 1
            elim_corr(ind_comp_list)
            elim_corr([row, col])
            # record the companies that have been categorized
            for i in ind_comp_list:
                set_num_counter[i] = record_dummy
            record_dummy -= 1

        # show the result
        for i in set(set_num_counter):
            self.cluster_dict[i] = set()
        for i in range(len(set_num_counter)):
            self.cluster_dict[set_num_counter[i]].add(self.company_list[i])

        new_sequence_num = [i for i in range(len(self.cluster_dict))]
        paper_class = [
            [(name, key) for name in item]
            for key, item in zip(new_sequence_num, self.cluster_dict.values())
        ]
        paper_class = functools.reduce(lambda x, y: x + y, paper_class)
        paper_class = pd.DataFrame(paper_class, columns=["company", "paper_class"])
        self.paper_class = paper_class

    def output_clusters_df(self):
        return self.paper_class

    def output_raw(self):
        return self.cluster_dict


class ML_cluster:
    def __init__(self, embedding, namelist):
        self.embedding = embedding
        self.namelist = namelist
        self.cluster_df = None
        self.name = None
        self.embedding_type=["kmean", "dbscan", "gmm"]
        self.how=["org", "cor", "cos"]

    def generate_clusters(
        self,
        embedding_type,
        how="org",
        n_clusters=10,
        random_state=None,
        DBSCAN_eps=0.5,
        DBSCAN_min_samples=5,
        DBSCAN_metric="cosine"
    ):
        assert how in ["org", "cor", "cos"]
        assert embedding_type in ["kmean", "dbscan", "gmm"]
        if how == "cor":
            train_matrix = np.corrcoef(self.embedding)
        elif how == "cos":
            train_matrix = cosine_similarity(self.embedding, dense_output=False)
        else:
            train_matrix = self.embedding

        if embedding_type == "kmean":
            # k-means model
            estimator = KMeans(n_clusters=n_clusters, random_state=random_state)
            estimator.fit(train_matrix)
            label_pred = estimator.labels_
            self.name = "kmean_class"
        elif embedding_type == "dbscan":
            # DBSCAN model
            estimator = DBSCAN(
                eps=DBSCAN_eps, min_samples=DBSCAN_min_samples, metric=DBSCAN_metric
            )
            estimator.fit(train_matrix)
            label_pred = estimator.labels_
            self.name = "DBSCAN_class"
        elif embedding_type == "gmm":
            # gmm model
            estimator = GaussianMixture(n_components=n_clusters,random_state=random_state)
            label_pred = estimator.fit_predict(train_matrix)
            self.name = "gmm_class"
        self.cluster_df = pd.DataFrame(
            zip([i for i in self.namelist], label_pred), columns=["company", self.name]
        )

    def output_clusters_df(self):
        return self.cluster_df

class get_all_cluster:
    def __init__(self,company_list,embadding_list):
        '''
        embadding_list: [embadding of first method, second method, ...]
        company_list and embadding of X method is a DataFrame or array. They Must have the same rank
        you can get embadding_list through combine_model.combine_embadding
        '''
        self.embadding_list=embadding_list
        self.company_list=company_list
        self.all_cluster=pd.DataFrame(company_list,columns=['company'])
    def get_all_cluster(self,
        n_cluster,
        random_state=None,
        use_DBSCAN=False,
        DBSCAN_eps=0.5,
        DBSCAN_min_samples=5,
        DBSCAN_metric="cosine"):
        i=0
        for embadding in tqdm(self.embadding_list,unit='per embadding',desc='handling...'):
            i+=1
            cluster=ML_cluster(embadding,self.company_list)
            for how in cluster.how:
                for method in cluster.embedding_type:
                    if use_DBSCAN and method=='dbscan':
                        cluster.generate_clusters(embedding_type=method,how=how,n_clusters=n_cluster)
                        df_i=cluster.output_clusters_df()
                        self.all_cluster[str(i)+'th embadding with '+how+' '+method]=df_i.iloc[:,-1]
                    elif method!='dbscan':
                        cluster.generate_clusters(embedding_type=method,how=how,n_clusters=n_cluster)
                        df_i=cluster.output_clusters_df()
                        self.all_cluster[str(i)+'th embadding with '+how+' '+method]=df_i.iloc[:,-1]
                    else:
                        continue
        return self.all_cluster

