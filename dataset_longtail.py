"""
    Title: movie_recommendation_using_KNN.ipynb
    Author: KevinLiao159
    Date: 31 Oct 2018
    Code version: 1.0
    Availability: https://github.com/KevinLiao159/MyDataSciencePortfolio/blob/master/movie_recommender/movie_recommendation_using_KNN.ipynb
    Comment: Code has been altered to fit purpose
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("initial_filter_step1.csv", na_values = ['no info', '.'], low_memory=False)

n_users = df.id.unique().shape[0]
n_items = df.Hotel_Address.unique().shape[0]
print('Number of users = ' + str(n_users) + ' | Number of hotels = ' + str(n_items))

# review frequency
df_ratings_cnt_tmp = pd.DataFrame(df.groupby('Reviewer_Score').size(), columns=['count'])
total_cnt = n_users * n_items
rating_zero_cnt = total_cnt - df.shape[0]
df_ratings_cnt = df_ratings_cnt_tmp.append(
    pd.DataFrame({'count': rating_zero_cnt}, index=[0.0]),
    verify_integrity=True,
).sort_index()

# hotel name frequency
df_hotels_cnt = pd.DataFrame(df.groupby('Hotel_Address').size(), columns=['count'])
# print df_hotels_cnt.head()

ap = df_hotels_cnt \
    .sort_values('count', ascending=False) \
    .reset_index(drop=True) \
    .plot(
        figsize=(12, 8),
        title='Rating Frequency of All hotels',
        fontsize=12
    )
ap.set_xlabel("Hotel Name")
ap.set_ylabel("number of ratings")
# plt.show(ap)

print df_hotels_cnt['count'].quantile(np.arange(1, 0.3, -0.05))


df_users_cnt = pd.DataFrame(df.groupby('id').size(), columns=['count'])
df_users_cnt.head()

# plot rating frequency of all hotels
ah = df_users_cnt \
    .sort_values('count', ascending=False) \
    .reset_index(drop=True) \
    .plot(
        figsize=(12, 8),
        title='Rating Frequency of All Users',
        fontsize=12
    )
ah.set_xlabel("user Id")
ah.set_ylabel("number of ratings")
plt.show(ah)

print df_users_cnt['count'].quantile(np.arange(1, 0.3, -0.05))
