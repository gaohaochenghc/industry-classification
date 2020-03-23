from sklearn.decomposition import PCA, TruncatedSVD
import pandas as pd
import copy
from combine_models import *


def compute_pca_svd(input_combine_models, cosine_matrix, type='pca'):
    assert type in ['pca', 'svd']
    combine_models=copy.deepcopy(input_combine_models)
    if type == 'pca':
        pca = PCA(n_components=2)
        reduced_cov = pd.DataFrame(pca.fit_transform(
            cosine_matrix), columns=['pca1', 'pca2'])
        combine_models.combine_class = pd.concat(
            [combine_models.combine_class, reduced_cov], axis=1)
        combine_models.combine_class[combine_models.namelist] = combine_models.combine_class[combine_models.namelist].astype(
            'category')

    elif type == 'svd':
        truncatedsvd = TruncatedSVD(n_components=2)
        reduced_cov = pd.DataFrame(truncatedsvd.fit_transform(
            cosine_matrix), columns=['svd1', 'svd2'])
        combine_models.combine_class = pd.concat(
            [combine_models.combine_class, reduced_cov], axis=1)
        combine_models.combine_class[combine_models.namelist] = combine_models.combine_class[combine_models.namelist].astype(
            'category')

    return combine_models


def trans_ticker(price_data):
    price_data.iloc[:,0]=price_data.iloc[:,0].map(lambda x:x[:-3]+'.SH' if x[0] in ['6','9'] else x[:-3]+'.SZ')
    return price_data
