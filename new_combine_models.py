import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#from scipy import stats
#from operator import itemgetter
from functools import reduce
from sklearn.linear_model import LinearRegression


class combine_models:
    """
    This class is used to combine different cluster results; 
    or combine different embadding matrix according to company name;
    or combine cluster results with embadding matrix( always used in performance_analysis)
    """

    def __init__(self):
        self.combine_cluster = None

    def get_combine_cluster(self, class_list):
        """Inputs:
                  class_list: a list contains different class_df. class_df has this columes: ['company','class_name']
           Notice: this class alway use inner join
        """
        if class_list != None:
            self.combine_cluster = class_list[0]
            for i in range(1, len(class_list)):
                temp_df = class_list[i].copy()
                self.combine_cluster = self.combine_cluster.merge(
                    temp_df, on=["company"], how="inner"
                )
            self.cluster_len = len(self.combine_cluster)
        return self.combine_cluster

    def get_combine_embadding(self, company_list, embadding_list):
        """Inputs:
                   embadding is a embadding list: [embadding1, embadding2, ...]
                   embaddingX has following columns: ['company', embadding matrix]
        """
        if embadding_list != None:
            combine_embadding = []
            company_list = pd.DataFrame(company_list, columns=["company"])
            for embadding in embadding_list:
                embadding = company_list.merge(
                    embadding, how="left", on=["company"]
                ).iloc[:, 1:]
                combine_embadding.append(embadding)
        return combine_embadding

    def get_merged_matrix(self, embadding, class_list):
        """Inputs:
                   embadding is a embadding list: [embadding1, embadding2, ...]
                   embaddingX has following columns: ['company', embadding matrix]
        """
        if self.combine_cluster == None:
            self.get_combine_cluster(class_list)
        new_embadding = []
        for i in range(len(embadding)):
            embadding[i] = embadding[i][
                embadding[i]["company"].isin(self.combine_cluster["company"])
            ]
            embadding[i]["company"].astype("category").cat.reorder_categories(
                self.combine_cluster["company"], inplace=True
            )
            new_embadding.append(
                embadding[i]
                .sort_values("company")
                .iloc[:, 1:]
                .set_index(pd.Index(range(self.cluster_len)))
            )
        return new_embadding


class performance_analysis:
    def __init__(self, class_df=None, return_df=None, embadding_list=None, market_return=None, factors=None):
        """Objective:
                use stock return to see the classification wellness

            Inputs:
                this class take class_df, return_df
                class_df is a dataframe with columns: ['company','class1','class2',...]
                return_df is a dataframe with columns: ['company','date','return']
                market_return is an optional dataframe with columns: ['date','market return']
                factors is an optional dataframe with columns: ['date', factor1, ...]
                date is a datetime object

        """

        self.class_df = class_df
        self.stock_return = return_df.merge(
            class_df, on=["company"], how="inner")
        self.company = list(set(class_df["company"]))
        self.method = list(class_df.columns[1:])
        self.company2int = {
            classname: dict(zip(class_df["company"], class_df[classname]))
            for classname in self.method
        }
        self.return_map = None
        self.statistical_map = None
        self.ret_df = None
        self.R2 = None
        self.has_mkt_ret = False
        self.has_factors = False
        self.mkt_ret = market_return
        self.factors = factors
        if not market_return is None:
            self.has_mkt_ret = True
            self.mkt_ret.columns = ['date', 'market_return']
        if not factors is None:
            self.has_factors = True

    def get_daliy_return(self, threshold=5, ret_comp=60):
        self.return_map = {}
        self.ret_df = {}
        pivot_df = self.stock_return[
            self.stock_return["company"].isin(self.class_df["company"])
        ].pivot(index="date", columns="company", values="return")
        pivot_df = pivot_df.dropna(thresh=pivot_df.shape[1]/2)
        class_df_trim = self.class_df[
            self.class_df["company"].isin(self.stock_return["company"])
        ]
        self.company_num = len(set(class_df_trim['company']))
        name_to_delete = []

        for name in self.method:
            ret_list = []
            self.return_map[name] = dict()
            cluster_namelist = set(class_df_trim[name].values.tolist())
            int2comp = {
                cl: class_df_trim[class_df_trim[name] == cl]["company"]
                for cl in cluster_namelist
            }
            if len(int2comp) == 1:
                self.return_map.pop(name, None)
                name_to_delete.append(name)
                continue
            for cl, comps in int2comp.items():
                if len(comps) > threshold:
                    self.return_map[name][cl] = pivot_df[comps]
                    self.return_map[name][cl].dropna(
                        1, thresh=ret_comp, inplace=True)
                    self.return_map[name][cl].fillna(0, inplace=True)
                    self.return_map[name][cl]["cluster_" + str(cl)] = self.return_map[name][
                        cl
                    ].mean(1)
                    app_df = self.return_map[name][cl]["cluster_" + str(cl)]
                    ret_list.append(app_df)
            self.ret_df[name] = reduce(
                lambda x, y: pd.merge(
                    x, y, how="outer", left_index=True, right_index=True
                ),
                ret_list,
            )
            self.ret_df[name].fillna(0, inplace=True)
            if self.has_mkt_ret:
                self.ret_df[name] = self.ret_df[name].merge(
                    self.mkt_ret, how='inner', on=['date'])
                for col in self.ret_df[name].columns[1:-1]:
                    self.ret_df[name][col] = self.ret_df[name][col] - \
                        self.ret_df[name].market_return
                if self.has_factors:
                    self.ret_df[name] = self.ret_df[name].merge(
                        self.factors, how='inner', on=['date'])

            #self.ret_df[name] = self.ret_df[name].set_index('date')

        for name in name_to_delete:
            self.method.remove(name)

    def get_statistical_describe(self):
        if self.return_map == None:
            self.get_daliy_return()
        self.statistical_map = {}
        self.R2 = {}
        for name in self.method:
            self.statistical_map[name] = {}
            self.R2[name] = pd.DataFrame(index=["R2"])
            class_num = len(self.return_map[name].keys())
            for classes in self.return_map[name].keys():
                self.statistical_map[name][classes] = pd.DataFrame(
                    index=self.return_map[name][classes].columns[:-1], columns=self.ret_df[name].columns)
                r2 = []
                for Xi in self.return_map[name][classes].columns[:-1]:
                    xy = self.ret_df[name].merge(
                        self.return_map[name][classes][Xi], how='inner', left_index=True, right_index=True)
                    reg = LinearRegression().fit(
                        xy.iloc[:, :-1], xy.iloc[:, -1])
                    r2.append(reg.score(xy.iloc[:, :-1], xy.iloc[:, -1]))
                    self.statistical_map[name][classes].loc[Xi, :] = reg.coef_
                self.statistical_map[name][classes]['most_positive_related'] = self.statistical_map[name][classes].iloc[:, :class_num].astype(
                    float).idxmax(axis=1)
                self.statistical_map[name][classes]['most_negative_related'] = (
                    -self.statistical_map[name][classes].iloc[:, :class_num].astype(float)).idxmax(axis=1)
                self.statistical_map[name][classes]['isBiggest'] = self.statistical_map[
                    name][classes]['most_positive_related'] == "cluster_" + str(classes)
                self.statistical_map[name][classes]['R2'] = r2
                self.statistical_map[name][classes].loc['summary',
                                                        'isBiggest'] = self.statistical_map[name][classes]['isBiggest'].sum()
                self.statistical_map[name][classes].loc['summary',
                                                        'R2'] = self.statistical_map[name][classes]['R2'].mean()
                self.R2[name][classes] = [np.mean(r2)]

    def print_table(self):
        table = pd.DataFrame(index=self.method, columns=[
                             'R2', 'proportion of right classification'])
        for i in table.index:
            table.loc[i, 'R2'] = self.R2[i].mean(1).iloc[0]
            right_classification = [
                self.statistical_map[i][j].loc['summary', 'isBiggest'] for j in self.return_map[i].keys()]
            table.loc[i, 'proportion of right classification'] = sum(
                right_classification)/self.company_num
            table.loc[i, 'classes number'] = int(len(right_classification))
            table.sort_values(
                by=['R2', 'proportion of right classification'], ascending=False, inplace=True)
        print('summary table:')
        return table
    
    def plot_industry_dense(self,rows = 6):
        fig, ax = plt.subplots(rows,len(self.method)//rows+1,figsize=(30,30))
        j=0
        k=0
        for i in self.method:
            sns.countplot(self.class_df[i], ax = ax[j,k])
            ax[j,k].set_title(i)
            j = j+1 if j<(rows-1) else 0
            k = k+1 if j==0 else k    