#!/usr/bin/env python
import os
cd = os.getcwd()
print cd
"""
    Title: Recommender system gui
    Author: Michelle Yip
    Date: 10 Apr 2019
    Code version: 1.0
"""
import numpy as np
import pandas as pd
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize, minmax_scale
import matplotlib
matplotlib.use('TkAgg')
import PySimpleGUI27 as sg

# read filtered file
train = pd.read_csv("initial_filter.csv", na_values = ['no info', '.'], low_memory=False)
# train = train.copy()
train['Rating_adjusted']= train.Reviewer_Score-train.User_Average_Score

# remove unpopular hotels
def remove_unpopular_hotels(df_hotels_cnt):
    df_hotels_cnt = pd.DataFrame(train.groupby('Hotel_Address').size(), columns=['count'])
    # must have at least 80 reviews
    popularity_thres = 80
    popular_hotels = list(set(df_hotels_cnt.query('count >= @popularity_thres').index))
    return train[train.Hotel_Address.isin(popular_hotels)]

# remove users with few ratings
def remove_small_users(df_users_cnt):
    df_users_cnt = pd.DataFrame(train.groupby('id').size(), columns=['count'])
    # filter data
    ratings_thres = 5
    active_users = list(set(df_users_cnt.query('count >= @ratings_thres').index))
    df_ratings_drop_users = train[train.id.isin(active_users)]
    return df_ratings_drop_users

# data filter function
def filter_test(type,train):
    if type == "remove hotels":
        return remove_unpopular_hotels(train)
    elif type == "remove users":
        return remove_small_users(train)
    elif type == "remove both":
        df_ratings_drop_hotels = remove_unpopular_hotels(train)
        return remove_small_users(df_ratings_drop_hotels)

# normalisation function
def norm_test(type,train):
    if type == "minmax":
        total_result = np.array(minmax_scale(train['Total_Number_of_Reviews'].astype(float).values.reshape(1,-1), axis=1).reshape(-1,1))
        train['norm_total'] = total_result

        pos_result = np.array(minmax_scale(train['Positive_Words_Percent'].astype(float).values.reshape(1,-1), axis=1).reshape(-1,1))
        train['norm_pos'] = pos_result

        date_result = np.array(minmax_scale(train['Review_Date'].astype(float).values.reshape(1,-1), axis=1).reshape(-1,1))
        train['norm_Date'] = date_result

    elif type == "l1 norm":
        total_result = normalize(train['Total_Number_of_Reviews'].astype(float).values.reshape(1,-1), norm='l1', axis=1).reshape(-1,1)
        result = [item[0] for item in total_result]
        train['norm_total'] = result

        pos_result = normalize(train['Positive_Words_Percent'].astype(float).values.reshape(1,-1), norm='l1', axis=1).reshape(-1,1)
        pos_res = [item[0] for item in pos_result]
        train['norm_pos'] = pos_res

        date_result = normalize(train['Review_Date'].astype(float).values.reshape(1,-1), norm='l1', axis=1).reshape(-1,1)
        date_res = [item[0] for item in date_result]
        train['norm_Date'] = date_res

    elif type == "l2 norm":
        total_result = normalize(train['Total_Number_of_Reviews'].astype(float).values.reshape(1,-1), norm='l2', axis=1).reshape(-1,1)
        result = [item[0] for item in total_result]
        train['norm_total'] = result

        pos_result = normalize(train['Total_Number_of_Reviews'].astype(float).values.reshape(1,-1), norm='l2', axis=1).reshape(-1,1)
        pos_res = [item[0] for item in pos_result]
        train['norm_pos'] = pos_res

        date_result = normalize(train['Review_Date'].astype(float).values.reshape(1,-1), norm='l2', axis=1).reshape(-1,1)
        date_res = [item[0] for item in date_result]
        train['norm_Date'] = date_res

    elif type == "z score":
        total_a = train['Total_Number_of_Reviews'].values.astype(float)
        total_result = (total_a-np.mean(total_a))/np.std(total_a)
        total_result.reshape(-1,1)
        train['norm_total'] = total_result

        total_pos = train['Positive_Words_Percent'].values.astype(float)
        pos_result = (total_pos-np.mean(total_pos))/np.std(total_pos)
        pos_result.reshape(-1,1)
        train['norm_pos'] = pos_result

        total_date = train['Review_Date'].values.astype(float)
        date_result = (total_date-np.mean(total_date))/np.std(total_date)
        date_result.reshape(-1,1)
        train['norm_Date'] = date_result
    return train

# feature tuning function
def feature_test(type, train, norm_type):
    if type == "ratings":
        if norm_type == 'none':
            train['Rate'] = train['Reviewer_Score']
        else:
            train['Rate'] = train['Rating_adjusted']
    elif type == "all features":
        if norm_type == 'none':
            train['Rate'] =train['Review_Date']+train['Total_Number_of_Reviews']+train['Rating_adjusted']+train['Positive_Words_Percent']
        else:
            train['Rate'] =train['Rating_adjusted']+train['norm_Date']+train['norm_total']+train['norm_pos']
    elif type == "no ratings":
        train['Rate'] =train['norm_Date']+train['norm_total']+train['norm_pos']
    elif type == "rating+date+total":
        train['Rate'] =train['Rating_adjusted']+train['norm_Date']+train['norm_total']
    elif type == "rating+date+word":
        train['Rate'] =train['Rating_adjusted']+train['norm_Date']+train['norm_pos']
    elif type == "rating+total+word":
        train['Rate'] =train['Rating_adjusted']+train['norm_total']+train['norm_pos']
    elif type == "rating+date":
        train['Rate'] =train['Rating_adjusted']+train['norm_Date']
    elif type == "rating+total":
        train['Rate'] =train['Rating_adjusted']+train['norm_total']
    elif type == "rating+word":
        train['Rate'] =train['Rating_adjusted']+train['norm_pos']
    elif type == "all normal":
        train['Rate'] =train['Review_Date']+train['Total_Number_of_Reviews']+train['Reviewer_Score']+train['Positive_Words_Percent']
    if 'Unnamed: 0' in train.columns:
        train.drop('Unnamed: 0', axis=1, inplace=True)
    if 'Unnamed: 1' in train.columns:
        train.drop('Unnamed: 1', axis=1, inplace=True)
    return train

# reducing the df size - only look at users which have visited a hotel in user's list
def hotel_df(train,userid,travelers,roomType,tripType):
    user1 = train[train['id']==userid]
    hotels_user = user1['Hotel_Address'].values #list of hotels rated by user
    other_users= train[train.Hotel_Address.isin(hotels_user)]
    tags = other_users['Tags'].values
    if travelers != "":
        if any(travelers in b for b in tags):
            other_users = other_users[other_users['Tags'].str.contains(travelers)]
    if roomType != "":
        if any(roomType in c for c in tags):
            other_users = other_users[other_users['Tags'].str.contains(roomType)]
    if tripType != "":
        if any(tripType in b for b in tags):
            other_users = other_users[other_users['Tags'].str.contains(tripType)]
    list = other_users.id.unique()
    list = list.astype(np.int64)
    return list

# pearson's correlation Method
def sim(train, userid1, userid2):
    user1 = train[train['id']==userid1]
    user2 = train[train['id']==userid2]
    if user1.empty: return 0
    if user2.empty: return 0

    hotel_df = hotel_List(train,user1,user2)

    top = 0
    btm_left = 0
    btm_right = 0
    btm = 0

    for hotel in hotel_df:
        u1_ratings = 0
        u2_ratings = 0
        if hotel in user1['Hotel_Address'].unique():
            u1_ratings = user1[user1.Hotel_Address==hotel].Rate.iloc[0]
        if hotel in user2['Hotel_Address'].unique():
            u2_ratings = user2[user2.Hotel_Address==hotel].Rate.iloc[0]
        top += u1_ratings*u2_ratings
        btm_left += pow(u1_ratings,2)
        btm_right += pow(u2_ratings,2)
    btm_left = math.sqrt(btm_left)
    btm_right = math.sqrt(btm_right)
    btm = btm_left*btm_right
    if btm == 0:
        return 0
    try: return top/btm
    except ZeroDivisionError: return 0

# returns a list of hotels both users have rated
def hotel_List(train,user1,user2):
    new = pd.concat([user1, user2], axis=0)
    return new.Hotel_Address.unique()

# filter users by country - returns list of users who have also visited the country
def country_filter(userList, country):
    newList = []
    for n,v in userList.iteritems():
        user1 = train[train['id']==n]
        hotels_user = user1['Hotel_Address'].values
        if any(country in s for s in hotels_user):
            newList.append(n)
    return newList

# the prediction method
def predict(userList,country):
    hotelList = pd.DataFrame()
    for n in userList:
        user1 = train[train['id']==n]
        hotels_user = user1['Hotel_Address'].values
        for hotel in hotels_user:
            if country in hotel:
                rate = user1.loc[(user1['Hotel_Address'] == hotel), 'Rate'].iat[0]
                hotelList = hotelList.append({'Hotel_Address': hotel, 'Rating': rate}, ignore_index=True)
    rating = hotelList.groupby('Hotel_Address')['Rating'].agg(['sum','count'])
    rating['New_Rate'] = rating['sum']/rating['count']
    rating.sort_values(by=['New_Rate'],inplace=True, ascending=False)
    rating.to_csv('rating_382.csv', encoding='utf-8')
    return rating.head(25)

# the user interface to allow user input
layout = [
          [sg.Text('Where would you like to fly to?')],
          [sg.Text('User ID', size=(22, 1), tooltip='Input your User ID'), sg.InputText('')],
          [sg.Text('Country/City Destination', size=(22, 1), tooltip='Where do you want to go?'), sg.InputText('')],
          [sg.Text('Type of Stay', size=(22, 1)), sg.InputCombo(('Leisure trip', 'Business trip'), size=(20, 3))],
          [sg.Text('Room Type', size=(22, 1)), sg.InputCombo(('Single Room', 'Twin Room', 'Double Room', 'Queen Room', 'King Room', 'Studio'), size=(20, 3))],
          [sg.Text('Travel Type', size=(22, 1)), sg.InputCombo(('Solo traveler', 'Couple', 'Group', 'Family with older children', 'Family with young children'), size=(20, 3))],
          [sg.Submit('Search'), sg.Cancel()]
         ]

window = sg.Window('Hotel Finder').Layout(layout)

# keep the window persistent
while True:
    button, values = window.Read()
    if button == 'Cancel':
        window.Close()
    elif button == 'Search':
        id_input = int(values[0])
        country_input = values[1]
        travel_input = values[2]
        stay_input = values[3]
        room_input = values[4]

        list_hotel = train['Hotel_Address'].values
        # validation
        if id_input not in train.id.values.tolist():
            sg.Popup("Invalid user Id")
        elif any(country_input in a for a in list_hotel) == False:
            sg.Popup("Invalid country/ city input")
        else:
            train['Positive_Words_Percent'].fillna(0, inplace=True)
            train['Review_Date'] = pd.to_datetime(train['Review_Date']).astype('int64')

            # selection of the best features to use in a recommender system
            train = norm_test('l2 norm',train)
            train = filter_test('remove both', train)
            train = feature_test('rating+date+total', train, 'l2 norm')

            hotelList = hotel_df(train,id_input,travel_input,room_input,travel_input)
            list1 = pd.Series(index=hotelList)
            for n in hotelList:
                pearson_corr = sim(train,id_input,n)
                print n
                list1[n] = pearson_corr
            # kNN - number of neighbours to consider
            top_sim = list1.sort_values(ascending=False).head(20)
            country = country_filter(top_sim,country_input)
            final = predict(country,country_input)
            hotel_rec = final.index
            hotel_score = final.New_Rate

            # create variables to print
            hotel_1 = "1. "+ hotel_rec[0] +" - "+ str(hotel_score[0])
            hotel_2 = "2. "+hotel_rec[1] +" - "+ str(hotel_score[1])
            hotel_3 = "3. "+hotel_rec[2] +" - "+ str(hotel_score[2])
            hotel_4 = "4. "+hotel_rec[3] +" - "+ str(hotel_score[3])
            hotel_5 = "5. "+hotel_rec[4] +" - "+ str(hotel_score[4])
            hotel_6 = "6. "+hotel_rec[5] +" - "+ str(hotel_score[5])
            hotel_7 = "7. "+hotel_rec[6] +" - "+ str(hotel_score[6])
            hotel_8 = "8. "+hotel_rec[7] +" - "+ str(hotel_score[7])
            hotel_9 = "9. "+hotel_rec[8] +" - "+ str(hotel_score[8])
            hotel_10 = "10. "+hotel_rec[9] +" - "+ str(hotel_score[9])
            hotel_11 = "11. "+hotel_rec[10] +" - "+ str(hotel_score[10])
            hotel_12 = "12. "+hotel_rec[11] +" - "+ str(hotel_score[11])
            hotel_13 = "13. "+hotel_rec[12] +" - "+ str(hotel_score[12])
            hotel_14 = "14. "+hotel_rec[13] +" - "+ str(hotel_score[13])
            hotel_15 = "15. "+hotel_rec[14] +" - "+ str(hotel_score[14])
            hotel_16 = "16. "+hotel_rec[15] +" - "+ str(hotel_score[15])
            hotel_17 = "17. "+hotel_rec[16] +" - "+ str(hotel_score[16])
            hotel_18 = "18. "+hotel_rec[17] +" - "+ str(hotel_score[17])
            hotel_19 = "19. "+hotel_rec[18] +" - "+ str(hotel_score[18])
            hotel_20 = "10. "+hotel_rec[19] +" - "+ str(hotel_score[19])
            hotel_21 = "21. "+hotel_rec[20] +" - "+ str(hotel_score[20])
            hotel_22 = "22. "+hotel_rec[21] +" - "+ str(hotel_score[21])
            hotel_23 = "23. "+hotel_rec[22] +" - "+ str(hotel_score[22])
            hotel_24 = "24. "+hotel_rec[23] +" - "+ str(hotel_score[23])
            hotel_25 = "25. "+hotel_rec[24] +" - "+ str(hotel_score[24])

            # create popup to show the top 25 recommendations to the user
            sg.Popup("Hotel Recommendations:",
                hotel_1,
                hotel_2,
                hotel_3,
                hotel_4,
                hotel_5,
                hotel_6,
                hotel_7,
                hotel_8,
                hotel_9,
                hotel_10,
                hotel_11,
                hotel_12,
                hotel_13,
                hotel_14,
                hotel_15,
                hotel_16,
                hotel_17,
                hotel_18,
                hotel_19,
                hotel_20,
                hotel_21,
                hotel_22,
                hotel_23,
                hotel_24,
                hotel_25,line_width=500)
