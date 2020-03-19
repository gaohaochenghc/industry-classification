import pandas as pd
import numpy as np
import functools
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.cluster import KMeans


class paper_sim:
    def __init__(self, company_list, company_vec):
        assert len(company_list) == company_vec.shape[0]
        self.company_list = company_list.copy()
        self.cosine_matrix = pairwise_distances(
            company_vec, metric="cosine", n_jobs=-1)
        self.cluster_dict = dict()
        self.name='paper_class'
        self.paper_class=None

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
        while len(set(set_num_counter)) > n_clusters:
            # get minimun correlation in the matrix
            min_pair = cov_matrix_enlarge.argmax()
            row = min_pair // cov_matrix_enlarge.shape[0]
            col = min_pair % cov_matrix_enlarge.shape[0]
            enlarge_company_list.append([row, col])
            if row == col:
                cov_matrix_enlarge[row, col] = 0
                continue
            # calculate the new generated industry correlation with other companies
            ind_comp_list = [i for i in query_ind(
                row)] + [i for i in query_ind(col)]
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
                        [sum_corr_func(i1, iter_company)
                         for i1 in ind_comp_list]
                    ) / len(ind_comp_list)
                else:
                    sum_corr = 0
                new_corr.append(sum_corr)
            cov_matrix_enlarge = np.vstack(
                (cov_matrix_enlarge, np.matrix(new_corr)))
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
        paper_class = functools.reduce(lambda x, y: x+y, paper_class)
        paper_class = pd.DataFrame(paper_class, columns=[
                                   "company", "paper_class"])

    def output_clusters_df(self):
        return self.paper_class

    def output_raw(self):
        return self.cluster_dict


class ML_clustering:

    def __init__(self,BERT_short,BERT_namelist,type,n_clusters=10):
        assert type in ['kmean']
        if type == 'kmean':
            # another comparable model: k-means model
            estimator=KMeans(n_clusters=n_clusters)
            estimator.fit(BERT_short)
            label_pred = estimator.labels_
            self.cluster_df = pd.DataFrame(zip([i for i in BERT_namelist], label_pred), columns=['company', 'kmean_class'])
            self.name='kmean_class'
            
    
    def output_clusters_df(self):
        return self.cluster_df