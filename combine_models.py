import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from operator import itemgetter
from sklearn.decomposition import PCA, TruncatedSVD


# combine the results of all clustering models


class combine_models:
    def __init__(self, class_list):
        '''Inputs:
                  class_list: a list contains different class_df. class_df has this columes: ['company','class_name']
           Notice: this class alway use inner join
        '''
        self.combine_df=class_list[0]
        for i in range(1,len(class_list)):
            temp_df=class_list[i].copy()
            self.combine_df=self.combine_df.merge(temp_df,on=['company'],how='inner')
        self.len=len(self.combine_df)

    def get_matrix(self,embadding):
        '''Inputs:
                   embadding is a embadding list: [embadding1, embadding2, ...]
                   embaddingX has following columns: ['company', embadding matrix]
        '''
        new_embadding=[]
        for i in range(len(embadding)):
            embadding[i]=embadding[i][embadding[i]['company'].isin(self.combine_df['company'])]
            embadding[i]['company'].astype('category').cat.reorder_categories(self.combine_df['company'],inplace=True)
            new_embadding.append(embadding[i].sort_values('company').iloc[:,1:].set_index(pd.Index(range(self.len))))
        return new_embadding



class performance_analysis:
    def __init__(self,mode,class_df=None,return_df=None,embadding_list=None):
        '''Inputs:
            This class has two mode: 
                1. apply dimension reduction method to see the performance of embadding matrix(word vector) and classification methods 
                2. use stock return to see the classification effect

                In the first mode:
                this class take class_df, embadding_list 
                class_df is a dataframe, which columns are: ['company','class1','class2',...]
                embadding_list: [embadding matrix of first method,embadding matrix of second method,...]
                
                In the second mode:
                this class take class_df, return_df
                class_df is a dataframe, which columns are: ['company','class1','class2',...]
                return_df is a dataframe, which columns are: ['company','date','return']
                date is a datetime object

            you can use mode=['matrix_performance','return_performance'] to switch types
        '''
        if mode=='matrix_performance':
            self.class_df=class_df
            self.embadding_list=embadding_list
            self.embadding_len=len(embadding_list) if embadding_list!=None else None
            self.class_len=class_df.shape[1]-1
        elif mode=='return_performance':
            self.stock_return=return_df.merge(class_df,on=['company'],how='inner')
            self.company=list(set(class_df['company']))
            self.method=class_df.columns[1:]
            self.company2int={classname:dict(zip(class_df['company'],class_df[classname])) for classname in self.method}
            self.return_map=None
            self.statistical_map=None
        else:
            raise ValueError('mode must in [matrix_performance, return_performance]')
    
    def plot_count_plot(self):
        fig, ax = plt.subplots(1,self.class_len,figsize=(20,6))
        plt.suptitle("Company number in different industry")
        for i in range(self.class_len):
            sns.countplot(self.class_df.iloc[:,i+1],ax=ax[i], order=self.class_df.iloc[:,i+1].value_counts().index)
            


    def get_reducted_vector(self,method='PCA'):
        pca=PCA(n_components=2)
        for matrix in self.embadding_list:
            reducted_vector=pd.DataFrame(pca.fit_transform(matrix))
            self.class_df=pd.concat([self.class_df,reducted_vector],axis=1)
    
    def plot_reducted_classification(self):
        self.class_df.iloc[:,1:]=self.class_df.iloc[:,1:].astype('category')
        if self.class_df.shape[1]==self.class_len+1:
            self.get_reducted_vector()
        fig, ax= plt.subplots(self.class_len,self.embadding_len,figsize=(20,20))
        for i in range(self.embadding_len):
            for j in range(self.class_len):
                sns.scatterplot(x=self.class_df.iloc[:,self.class_len+1+2*i],y=self.class_df.iloc[:,self.class_len+2+2*i],ax=ax[j,i],hue=self.class_df.iloc[:,j+1])
                ax[j,i].set_title(self.class_df.columns[j+1]+' with the '+str(i)+'th embadding')


    def get_daliy_return(self):
        self.return_map={}
        for name in self.method:
            mean_return=self.stock_return.groupby(by=[name,'date']).mean()['return']
            self.return_map[name]={clas:pd.DataFrame(mean_return[clas,]).dropna() for clas in mean_return.index.levels[0]}
            for company in self.company:
                ind=self.company2int[name][company]
                company_return=self.stock_return[self.stock_return['company']==company][['date','return']].set_index('date').rename(columns={'return':company})
                if company_return.shape[0]==0:
                    continue
                self.return_map[name][ind]=self.return_map[name][ind].join(company_return)

        
   
    def get_statistical_discribe(self):
        if self.return_map==None:
            self.get_daliy_return()
        self.statistical_map={}
        for name in self.method:
            self.statistical_map[name]={key:pd.DataFrame(columns=item.columns,index=['abs_slope','p','unsignificance_num','R2']) for key, item in self.return_map[name].items()}
            for classes in self.statistical_map[name]:
                reg=[itemgetter(0,2,3)(stats.linregress(self.return_map[name][classes][j][self.return_map[name][classes][j].isna()==False],self.return_map[name][classes]['return'][self.return_map[name][classes][j].isna()==False])) for j in self.return_map[name][classes].columns]
                self.statistical_map[name][classes].loc['unsignificance_num']=list(map(lambda x:x[2]<0.001,reg))
                self.statistical_map[name][classes].loc['p']=list(map(lambda x: x[2],reg))
                self.statistical_map[name][classes].loc['R2']=list(map(lambda x:x[1]**2,reg))
                self.statistical_map[name][classes].loc['abs_slope']=list(map(lambda x:abs(x[0]),reg))
                self.statistical_map[name][classes]['sum'+str(classes)]=self.statistical_map[name][classes].mean(axis=1)
                self.statistical_map[name][classes].loc['unsignificance_num','sum'+str(classes)]=sum(self.statistical_map[name][classes].loc['unsignificance_num',]==False)
                self.statistical_map[name][classes].drop(columns='return',inplace=True)
        

    def get_wellness(self):
        if self.statistical_map==None:
            self.get_statistical_discribe()
        self.wellness_map={name:pd.DataFrame(index=['abs_slope','p','unsignificance_num','R2']) for name in self.method}
        for name in self.wellness_map:
            for value in self.statistical_map[name].values():
                 self.wellness_map[name]=self.wellness_map[name].join(value.iloc[:,-1])
            self.wellness_map[name]['mean']=self.wellness_map[name].mean(axis=1)
            self.wellness_map[name].loc['unsignificance_num','mean']=sum(self.wellness_map[name].iloc[2,:-1])
        

    def output(self,name='wellness'):
        if name=='daliy_return':
            return self.return_map
        elif name=='discribe':
            return self.statistical_map
        elif name=='wellness':
            return self.wellness_map

    def get_regression_plot(self):
        for name in self.wellness_map:
            print('Mean of R square of '+name+' classification is: ',self.wellness_map[name].loc['R2','mean'])
        methods=len(self.method)
        fig, ax=plt.subplots(3,methods,figsize=(20,20))
        i=0  
        color=['grey','red','blue','purple','green','yellow','black','grey']
        for name in self.wellness_map:
            sns.distplot(self.wellness_map[name].iloc[0,:-1],ax=ax[0,i],color=color[i])
            ax[0,i].set_title('slope of '+name+' classification')
            sns.distplot(self.wellness_map[name].iloc[3,:-1],ax=ax[1,i],color=color[i])
            ax[1,i].set_title('R sqpare of '+name+' classification')
            sns.barplot(x=self.wellness_map[name].columns[:-1],y=self.wellness_map[name].iloc[1,:-1],ax=ax[2,i],color=color[i])
            ax[2,i].set_title('p value of '+name+' classification')
            i+=1
    
