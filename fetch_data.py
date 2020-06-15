# -*- coding: utf-8 -*-
"""
Created on Sun May 24 13:35:56 2020

@author: Gao haocheng
"""


import pandas as pd
import numpy as np
import time
from bert_serving.client import BertClient
from tqdm import tqdm
from tqdm.notebook import tqdm, trange
from selenium import webdriver
from lxml import etree


class fetch_data:
    '''
    fetch data from Yahoo. You can choose to fetch the description data or financial statement data
    '''
    def __init__(self, company):
        self.company = company
        
    def init_driver(self):
        #options = webdriver.firefox.options.Options()
        
        options = webdriver.ChromeOptions()
    
        options.add_argument("-headless")
        options.add_argument('--ignore-certificate-errors')
    
        #driver = webdriver.Firefox(options=options)
        
        self.driver = webdriver.Chrome(options=options)
        
    def get_single_statement(self, ticker):
        
        url='https://finance.yahoo.com/quote/'+ticker+'/key-statistics?p='+ticker
    
        self.driver.get(url)
        
        time.sleep(3)
    
        html = self.driver.page_source
        html = etree.HTML(html)
    
        mkt_cap = html.xpath('//*[@id="Col1-0-KeyStatistics-Proxy"]/section/div[3]/div[1]/div[2]/div/div[1]/div[1]/table/tbody/tr[1]/td[3]//text()')[0]
        pb_ratio = html.xpath('//*[@id="Col1-0-KeyStatistics-Proxy"]/section/div[3]/div[1]/div[2]/div/div[1]/div[1]/table/tbody/tr[7]/td[3]//text()')[0]
        beta = html.xpath('//*[@id="Col1-0-KeyStatistics-Proxy"]/section/div[3]/div[2]/div/div[1]/div/div/table/tbody/tr[1]/td[2]//text()')[0]
    
        profit_m = html.xpath('//*[@id="Col1-0-KeyStatistics-Proxy"]/section/div[3]/div[3]/div/div[2]/div/div/table/tbody/tr[1]/td[2]//text()')[0]
        roa = html.xpath('//*[@id="Col1-0-KeyStatistics-Proxy"]/section/div[3]/div[3]/div/div[3]/div/div/table/tbody/tr[1]/td[2]//text()')[0]
        roe = html.xpath('//*[@id="Col1-0-KeyStatistics-Proxy"]/section/div[3]/div[3]/div/div[3]/div/div/table/tbody/tr[2]/td[2]//text()')[0]
    
        return [ticker,mkt_cap,pb_ratio, beta, profit_m, roa, roe]
    
    def get_statement_data(self):
        col=['company','mkt_cap','pb_ratio','beta','profit_m','roa','roe']
        self.ratio_df=pd.DataFrame(columns=col)
        self.init_driver()
        
        try_num=0
        single_iter=0
        while True:
            for t in trange(try_num, len(self.company)):
                try:
                    self.ratio_df=self.ratio_df.append(pd.DataFrame([self.get_single_statement(self.company[t])],columns=col),ignore_index=True)
                    single_iter=0
                except:
                    self.driver.close()
                    single_iter+=1
                    time.sleep(10)
                    self.init_driver()
                    if single_iter>=5:
                        single_iter=0
                        continue
                    try_num=t
                    break
            else:
                break
        return self.ratio_df
    
    def get_single_des(self, ticker):
        url = 'https://finance.yahoo.com/quote/'+ticker+'/profile?p='+ticker

        self.driver.get(url)

        time.sleep(2)
        html = self.driver.page_source
        html = etree.HTML(html)
    
        # page = requests.get(url)
        # html = etree.HTML(page.text)
        
        # items = html.xpath('//*[@id="Col1-0-Profile-Proxy"]/section/section[2]/p')
        intro = html.xpath(
            '//*[@id="Col1-0-Profile-Proxy"]/section/section[2]/descendant-or-self::text()')
        segment = html.xpath(
            '//*[@id="Col1-0-Profile-Proxy"]/section/div[1]/div/div/p[2]/span[2]//text()')
        industry = html.xpath(
            '//*[@id="Col1-0-Profile-Proxy"]/section/div[1]/div/div/p[2]/span[4]//text()')

        return [ticker,intro,segment,industry]
            
    def get_description_data(self):
        col=['company','intro','segment','industry']
        
        self.des_df=pd.DataFrame(columns=col)
        self.init_driver()
        
        try_num=0
        single_iter=0
        while True:
            for t in trange(try_num, len(self.company)):
                try:
                    self.des_df=self.des_df.append(pd.DataFrame([self.get_single_des(self.company[t])],columns=col),ignore_index=True)
                    single_iter=0
                except:
                    self.driver.close()
                    single_iter+=1
                    time.sleep(10)
                    self.init_driver()
                    if single_iter>=5:
                        single_iter=0
                        continue
                    try_num=t
                    break
            else:
                break
        return self.des_df
    
    
def get_bert(des):
    des['len'] = des['intro'].str.len()
    des.set_index('company', inplace=True)
    
    short = des[des['intro'].str.len()<=512]
    long = des[(des['intro'].str.len()>512) & (des['intro'].str.len()<1024)]
    # max length of bert is 512
     
    long_first_part = long['intro'].str[:512]
    long_second_part = long['intro'].str[512:]
    long_second_part = long_second_part[long_second_part.str.len()>100]
     
    short_intro=short['intro'].values.tolist()
    long_first_part_intro = long_first_part.values.tolist()
    long_second_part_intro = long_second_part.values.tolist()
         
    bc = BertClient()
    short_embadding = bc.encode(short_intro)
    long_first_part_embadding = bc.encode(long_first_part_intro)
    long_second_part_embadding = bc.encode(long_second_part_intro)
     
    short_embadding = pd.DataFrame(short_embadding, index = short_intro.index)
    long_first_part_embadding = pd.DataFrame(long_first_part_embadding, index = long_first_part.index)
    long_second_part_embadding = pd.DataFrame(long_second_part_embadding, index = long_second_part.index)
     
    temp = long_first_part_embadding.reindex(long_second_part_embadding.index)
    temp = (temp + long_second_part_embadding)/2
     
    long_first_part_embadding.loc[temp.index,:] = temp
 
    return pd.concat([short_embadding, long_first_part_embadding])
     
        
        
        
        
            
            
            