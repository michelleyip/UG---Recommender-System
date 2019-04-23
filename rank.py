"""
    Title: Kendall Tau and Spearman rank calculations
    Author: Michelle Yip
    Date: 10 Apr 2019
    Code version: 1.0
"""
import numpy as np
import pandas as pd
import math
import scipy.stats as stats

# edit csv file names below
rating_list = pd.read_csv("rating_45NN.csv", na_values = ['no info', '.'], low_memory=False).head(100)
no_rating_list = pd.read_csv("rating_20NN.csv", na_values = ['no info', '.'], low_memory=False).head(100)
x =rating_list['Hotel_Address'].values
y =no_rating_list['Hotel_Address'].values

# kendall tau
tau, p_value = stats.kendalltau(x, y)
print 'tau ', tau
print 'p ', p_value

# spearman rank
spearman = stats.spearmanr(x, y)[0]
print 'spearman ', spearman
