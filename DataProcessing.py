import os
import pandas as pd
import math
import gzip
import numpy as np
import scipy.stats as stats
from scipy.sparse import csc_matrix, csr_matrix, coo_matrix
import tensorflow as tf
import csv

def getBins(x, bin_count = 30):
    bins = [-1]
    binvalue = x // bin_count +1
    for i in range(bin_count):
        bins.append(binvalue*(i+1))
    return bins

def train_test_split(ratings, num_reviews = 2, time_bin_set = None, time_day_set = None):
    test = np.zeros(ratings.shape)
    train = ratings.copy()
    #train_bin_set = None
    #test_bin_set = None
    #train_td_set = None
    #test_td_set = None
    if(time_bin_set is not None):
        train_bin_set = time_bin_set.copy()
        test_bin_set = np.zeros(time_bin_set.shape)
    if(time_day_set is not None):
        train_td_set = time_day_set.copy()
        test_td_set = np.zeros(time_day_set.shape)
        
    for user in range(ratings.shape[0]):
        test_ratings = np.random.choice(ratings[user, :].nonzero()[0], 
                                        size=num_reviews, 
                                        replace=False)
        train[user, test_ratings] = 0.
        test[user, test_ratings] = ratings[user, test_ratings]
        
        if(time_bin_set is not None):
            train_bin_set[user, test_ratings] = 0
            test_bin_set[user, test_ratings] = time_bin_set[user, test_ratings]
            
        if(time_day_set is not None):
            train_td_set[user, test_ratings] = 0
            test_td_set[user, test_ratings] = time_day_set[user, test_ratings]
            
    # Test and training are truly disjoint
    assert(np.all((train * test) == 0)) 
    assert(np.all((train_bin_set * test_bin_set) ==0))               
    assert(np.all((train_td_set * test_td_set) ==0))   
           
    return train, test, train_bin_set, test_bin_set, train_td_set, test_td_set

def PreProcessAmazonDF(df, bin_count = 30):
    
    df['unixReviewTime'] = pd.to_datetime(df['unixReviewTime'],unit='s') #changing time format
    df.rename(columns={'unixReviewTime': 'ReviewTime', 'asin': 'product'}, inplace=True)
    df = df.drop('reviewTime',1)
    df['reviewerID'] = df['reviewerID'].astype("category")
    df['product'] = df['product'].astype("category")

    df['zeroday'] = pd.Timestamp('1996-01-01')
    df['ReviewDay'] = (df['ReviewTime'] - df['zeroday'] ).dt.days
    df["ReviewDay"] = df["ReviewDay"]-df["ReviewDay"].min()
    df = df.drop('zeroday', axis = 1)

    bins = getBins(df["ReviewDay"].max(), bin_count)
    group_value_names = range(bin_count)
    df['ITBin'] = pd.cut(df['ReviewDay'], bins, labels=group_value_names)
    df['ITBin'] = df['ITBin'].astype("int")
    
    df["userID"] = df['reviewerID'].cat.codes
    df["itemID"] = df["product"].cat.codes
    
    df["TDayCat"] = df["ReviewDay"].astype("category").cat.codes.values
    return df

def convert_csr_to_sparse_tensor_inputs(X):
    coo = X.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return indices, coo, coo.shape 

def convert_to_sparse_tensor_inputs(X):
    coo = coo_matrix(X)
    indices = np.mat([coo.row, coo.col]).transpose()
    return indices, coo, coo.shape 

def get_base_loss(train_set, dev_set, test_set):
    mean_set = sum(train_set["coo"].data)/len(train_set["coo"].data)
    print( "mean is:", mean_set)
    return (sum((train_set["coo"].data - mean_set)**2)/len(train_set["coo"].data), sum((dev_set["coo"].data - mean_set)**2)/len(dev_set["coo"].data),sum((test_set["coo"].data - mean_set)**2)/len(test_set["coo"].data))

def get_df_base_loss(train_df, dev_df, test_df):
    mean_set = train_df["overall"].mean()
    print("mean is: ", mean_set)
    return (sum((train_df["overall"].values - mean_set)**2) / train_df["overall"].count(),  
            sum((dev_df["overall"].values - mean_set)**2) / dev_df["overall"].count(), 
            sum((test_df["overall"].values - mean_set)**2) / test_df["overall"].count())
           
def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield eval(l)

def getDF(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')           

def getcsvDF(path):
    i = 0
    df = {}
    with open(path) as f:
        readers = csv.reader(f)
        for d in readers:
            df[i] = d
            i += 1
    return pd.DataFrame.from_dict(df, orient='index')

def getMeanDay(dt):
    csrdt = csr_matrix(dt)
    sums = csrdt.sum(axis=1).A1
    counts = np.diff(csrdt.indptr)
    averages = sums / counts
    return averages
def get_dense_from_csv(filename):
    return np.genfromtxt(filename, delimiter=',', dtype=float, skip_header = 1)[:, 1:]

def load_from_clean_csv(type_str):
    train_rank_set= get_dense_from_csv("data\\" + type_str + "_train_rating.csv")
    dev_rank_set = get_dense_from_csv("data\\" + type_str + "_dev_rating.csv")
    test_rank_set = get_dense_from_csv("data\\" + type_str + "_test_rating.csv")
    
    train_bin_set= get_dense_from_csv("data\\" + type_str + "_train_bin.csv")
    dev_bin_set = get_dense_from_csv("data\\" + type_str + "_dev_bin.csv")
    test_bin_set = get_dense_from_csv("data\\" + type_str + "_test_bin.csv")
    
    train_td_set= get_dense_from_csv("data\\" + type_str + "_train_day.csv")
    dev_td_set = get_dense_from_csv("data\\" + type_str + "_dev_day.csv")
    test_td_set = get_dense_from_csv("data\\" + type_str + "_test_day.csv")
    
    train_set = {}
    train_set["indices"], train_set["coo"], train_set["shape"] = convert_to_sparse_tensor_inputs(train_rank_set)
    train_set["dense_tbin"] = train_bin_set
    train_set["dense_tday"] = train_td_set
    dev_set = {}
    dev_set["indices"], dev_set["coo"], dev_set["shape"] = convert_to_sparse_tensor_inputs(dev_rank_set)
    dev_set["dense_tbin"] = dev_bin_set
    dev_set["dense_tday"] = dev_td_set
    test_set ={}
    test_set["indices"], test_set["coo"], test_set["shape"] = convert_to_sparse_tensor_inputs(test_rank_set)
    test_set["dense_tbin"] = test_bin_set
    test_set["dense_tday"] = test_td_set
    
    return train_set, dev_set, test_set

def load_from_raw_df(load_filename, save_filename,  bin_count):
    df = getDF('data\\reviews_Beauty_10.json.gz')
    df = PreProcessAmazonDF(df, bin_count)
    reviews_matrix = csr_matrix((df['overall'].astype(float), 
                   (df['reviewerID'].cat.codes,
                    df['product'].cat.codes 
                    ))) 

    tbin_matrix = csr_matrix((df['ITBin'].astype(int), 
                       (df['reviewerID'].cat.codes,
                        df['product'].cat.codes 
                        ))) 
    tday_matrix = csr_matrix((df['ReviewDay'].astype(int), 
                       (df['reviewerID'].cat.codes,
                        df['product'].cat.codes 
                        ))) 
    train_rank_set, test_rank_set, train_bin_set, test_bin_set, train_td_set, test_td_set = train_test_split(reviews_matrix.toarray(), num_reviews = 2, time_bin_set = tbin_matrix.toarray(), time_day_set = tday_matrix.toarray())
    train_rank_set, dev_rank_set, train_bin_set, dev_bin_set, train_td_set, dev_td_set = train_test_split(train_rank_set, num_reviews = 2, time_bin_set = train_bin_set, time_day_set = train_td_set)

    train_set = {}
    train_set["indices"], train_set["coo"], train_set["shape"] = convert_to_sparse_tensor_inputs(train_rank_set)
    train_set["mean_rank"] = np.mean(train_set["coo"].data)
    train_set["dense_tbin"] = train_bin_set
    train_set["dense_tday"] = train_td_set
    
    train_set["mean_day"] = getMeanDay(train_set["dense_tday"])

    
    #print(train_set)
    dev_set = {}
    dev_set["indices"], dev_set["coo"], dev_set["shape"] = convert_to_sparse_tensor_inputs(dev_rank_set)
    dev_set["dense_tbin"] = dev_bin_set
    dev_set["dense_tday"] = dev_td_set
    
    test_set ={}
    test_set["indices"], test_set["coo"], test_set["shape"] = convert_to_sparse_tensor_inputs(test_rank_set)
    test_set["dense_tbin"] = test_bin_set
    test_set["dense_tday"] = test_td_set 
    
    pd.DataFrame(train_set["coo"].toarray()).to_csv('data\\'+ save_filename + '_train_rating.csv')
    pd.DataFrame(dev_set["coo"].toarray()).to_csv('data\\'+ save_filename + '_dev_rating.csv')
    pd.DataFrame(test_set["coo"].toarray()).to_csv('data\\'+ save_filename + '_test_rating.csv')

    pd.DataFrame(train_set["dense_tbin"]).to_csv('data\\'+ save_filename + '_train_bin.csv')
    pd.DataFrame(dev_set["dense_tbin"]).to_csv('data\\'+ save_filename + '_dev_bin.csv')
    pd.DataFrame(test_set["dense_tbin"]).to_csv('data\\'+ save_filename + '_test_bin.csv')

    pd.DataFrame(train_set["dense_tday"]).to_csv('data\\'+ save_filename + '_train_day.csv')
    pd.DataFrame(dev_set["dense_tday"]).to_csv('data\\'+ save_filename + '_dev_day.csv')
    pd.DataFrame(test_set["dense_tday"]).to_csv('data\\'+ save_filename + '_test_day.csv')
    
    print("Loaded file successfully and saved clean data to disk!")
    
    return train_set, dev_set, test_set

def getMeanDaybyUser(df):
    mean_u_day = df.groupby("userID")["ReviewDay"].agg({"mean"}) #["mean"] #get the average user review date
    mean_u_day = mean_u_day.reset_index(level=2, drop = True)
    return mean_u_day["mean"].values
def getUserRatedItemCount(df):
    uri_count = df.groupby("userID").agg({"itemID": lambda x: x.nunique()})
    uri_count = uri_count.reset_index(level=2, drop = True)
    return uri_count["itemID"].values
def getUserRatedItemCountNonUnique(df):
    uri_count = df.groupby("userID").count()
    uri_count = uri_count.reset_index(level=2, drop = True)
    return uri_count["itemID"].values

def patch_with_value(x, patch_value, max_length):
    new_list = np.full(max_length, patch_value, dtype=int)
    new_list[:len(x)] = x
    return new_list

def getImplicitDF(df): 
    max_item_id = max(df["itemID"].tolist())

    item_len_df = df.groupby("userID").agg({"itemID": lambda x: len(x)}) 
    item_len_df = item_len_df.reset_index(level=2, drop = True)

    user_item_df = df.groupby("userID")["itemID"].agg({"itemList": lambda x:  tuple(x)}) 
    user_item_df = user_item_df.reset_index(level=2, drop = True)
    maxlength = max(len(x) for x in user_item_df["itemList"].tolist())
    user_item_df["itemList"] = user_item_df["itemList"].apply(lambda x: patch_with_value(x, max_item_id + 1, maxlength))
    user_item_df["itemLen"] = item_len_df["itemID"]
    
    return user_item_df
