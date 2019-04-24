"""
    Title: Initial Filter
    Author: Michelle Yip
    Date: 10 Apr 2019
    Code version: 1.0
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# open original csv file
df = pd.read_csv("Hotel_Reviews.csv", na_values = ['no info', '.'], low_memory=False)

#assigns unique id for every user
df['id'] = df.groupby(['Reviewer_Nationality','Total_Number_of_Reviews_Reviewer_Has_Given'], sort=False).ngroup()

# calculate Total number of reviews per hotel
total_hotels = df.groupby(['Hotel_Address']).Hotel_Address.agg('count')
df['Total_Number_of_Reviews'] = df['Hotel_Address']
df['Total_Number_of_Reviews'] = df['Total_Number_of_Reviews'].map(total_hotels.to_dict())

# calculate total number of reviews by user
total_user = df.groupby(['id']).id.agg('count')
df['Total_Number_of_Reviews_Reviewer_Has_Given'] = df['id']
df['Total_Number_of_Reviews_Reviewer_Has_Given'] = df['Total_Number_of_Reviews_Reviewer_Has_Given'].map(total_user.to_dict())
df.drop(df[df.Total_Number_of_Reviews_Reviewer_Has_Given > 1400].index, inplace=True)

# sort by date
df['Review_Date'] =pd.to_datetime(df.Review_Date)
df.sort_values(by=['Review_Date'], inplace=True)

# drops the oldest reviews where id and hotel review is the same
df.drop_duplicates(subset=['Hotel_Address','id'], keep='last', inplace=True)

# recalculate the average scoring for each hotel and round to 1dp
hotel_avg = df.groupby('Hotel_Address')['Reviewer_Score'].mean()
df['Average_Score'] = df['Hotel_Address']
df['Average_Score'] = df['Average_Score'].map(hotel_avg.to_dict())
# df.Average_Score = df.Average_Score.round(1)

# calculate the average scoring for each user and round to 1dp
hotel_avg = df.groupby('id')['Reviewer_Score'].mean()
df['User_Average_Score'] = df['id']
df['User_Average_Score'] = df['User_Average_Score'].map(hotel_avg.to_dict())
# df.User_Average_Score = df.User_Average_Score.round(1)

# calculate %positive words
df['total_words'] = df['Review_Total_Negative_Word_Counts']+df['Review_Total_Positive_Word_Counts']
df['Positive_Words_Percent'] = (df['Review_Total_Positive_Word_Counts']/df['total_words'])
df['Positive_Words_Percent'].fillna(0, inplace=True)

df['Review_Date'] = pd.to_datetime(df['Review_Date']).astype('int64')

# remove columns
df.drop(['Additional_Number_of_Scoring'], axis=1, inplace=True)
df.drop(['Review_Total_Negative_Word_Counts'], axis=1, inplace=True)
df.drop(['Review_Total_Positive_Word_Counts'], axis=1, inplace=True)

# saves new file
df.to_csv('initial_filter.csv', encoding='utf-8')
