"""
    Title: Train/Test Split
    Author: Michelle Yip
    Date: 10 Apr 2019
    Code version: 1.0
    Comment: On average takes 3 minutes to run
"""
import numpy as np
import pandas as pd
import math
from sklearn.preprocessing import normalize, minmax_scale
import matplotlib
matplotlib.use('TkAgg')
import PySimpleGUI27 as sg

# remove unpopular hotels
def remove_unpopular_hotels(df_hotels_cnt):
    df_hotels_cnt = pd.DataFrame(train.groupby('Hotel_Address').size(), columns=['count'])
    # must have at least 80 reviews per hotel
    popularity_thres = 80
    popular_hotels = list(set(df_hotels_cnt.query('count >= @popularity_thres').index))
    return train[train.Hotel_Address.isin(popular_hotels)]

# remove users with few ratings
def remove_small_users(df_users_cnt):
    df_users_cnt = pd.DataFrame(train.groupby('id').size(), columns=['count'])
    # must have at least 5 reviews per user
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
    # print "user_list"
    # print len(list)
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
            u1_ratings = user1[user1.Hotel_Address==hotel].Rating_adjusted.iloc[0]
        if hotel in user2['Hotel_Address'].unique():
            u2_ratings = user2[user2.Hotel_Address==hotel].Rating_adjusted.iloc[0]
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
    rating = hotelList.groupby('Hotel_Address')['Rating'].agg(['sum','count'])  #fix??
    rating['New_Rate'] = rating['sum']/rating['count']
    rating.sort_values(by=['New_Rate'],inplace=True, ascending=False)
    rating.to_csv('rating_382.csv', encoding='utf-8')
    return rating.head(20)

# return recall
def get_recall(final,user_hotel):
    final_hotel = final.Hotel_Address.unique()
    true_p = set(user_hotel) & set(final_hotel)
    true_p1 = len(true_p)
    btm = len(user_hotel)
    recall = true_p1/btm
    return recall

# return precision
def get_precision(final,user_hotel):
    final_hotel = final.Hotel_Address.unique()
    true_p = set(user_hotel) & set(final_hotel)
    true_p1 = len(true_p)
    btm = len(final_hotel)
    recall = true_p1/btm
    return recall

# return f1
def get_f1(recall,precision):
    top = recall*precision
    btm = recall+precision
    try: return 2*(top/btm)
    except ZeroDivisionError: return np.nan

# return mae
def get_mae(final,test):
    user_hotel = test.Hotel_Address.unique()
    final_hotel = final.Hotel_Address.unique()
    common = set(user_hotel) & set(final_hotel)
    if common == set():
        return np.nan
    sum = 0
    count = 0
    for hotel in common:
        count+=1
        predicted = final.loc[(final['Hotel_Address'] == hotel), 'New_Rate'].iat[0]
        actual = test.loc[(test['Hotel_Address'] == hotel), 'Rate'].iat[0]
        sum+= abs(predicted-actual)
    mae = sum/count
    return round(mae,6)

# return rmse
def get_rmse(final,test):
    user_hotel = test.Hotel_Address.unique()
    final_hotel = final.Hotel_Address.unique()
    common = set(user_hotel) & set(final_hotel)
    if common == set():
        return np.nan
    sum = 0
    count = 0
    for hotel in common:
        count+=1
        predicted = final.loc[(final['Hotel_Address'] == hotel), 'New_Rate'].iat[0]
        actual = test.loc[(test['Hotel_Address'] == hotel), 'Rate'].iat[0]
        sum+= math.pow(predicted-actual,2)
    intermediate = sum/count
    rmse = math.sqrt(intermediate)
    return round(rmse,6)

# the user interface to allow user input
layout = [
          [sg.Text('Where would you like to fly to?')],
          [sg.Text('User ID', size=(22, 1), tooltip='Input your User ID'), sg.InputText('')],
          [sg.Text('Country/City Destination', size=(22, 1), tooltip='Where do you want to go?'), sg.InputText('')],
          [sg.Text('Travel Type', size=(22, 1)), sg.InputCombo(('Solo traveler', 'Couple', 'Group', 'Family with older children', 'Family with young children'), size=(20, 3))],
          [sg.Text('Type of Stay', size=(22, 1)), sg.InputCombo(('Leisure trip', 'Business trip'), size=(20, 3))],
          [sg.Text('Room Type', size=(22, 1)), sg.InputCombo(('Single Room', 'Twin Room', 'Double Room', 'Queen Room', 'King Room', 'Studio'), size=(20, 3))],
          [sg.Text(' ', size=(22, 1))],

          [sg.Text('Select Norm', size=(22, 1)), sg.InputCombo(('none', 'minmax', 'l1 norm', 'l2 norm', 'z score'), size=(20, 3))],
          [sg.Text('Select Filter', size=(22, 1)), sg.InputCombo(('none', 'remove hotels', 'remove users', 'remove both'), size=(20, 3))],
          [sg.Text('Select Feature', size=(22, 1)), sg.InputCombo(('ratings', 'all features', 'no ratings', 'rating+date+total', 'rating+date+word', 'rating+total+word','rating+date','rating+total','rating+word','all normal'), size=(20, 3))],

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
        norm_input = values[5]
        filter_input = values[6]
        feature_input = values[7]

        train = pd.read_csv("1_filter_train.csv", na_values = ['no info', '.'], low_memory=False)
        train['Rating_adjusted']= train.Reviewer_Score-train.User_Average_Score
        train['Positive_Words_Percent'].fillna(0, inplace=True)
        train['Review_Date'] = pd.to_datetime(train['Review_Date']).astype('int64')

        list_hotel = train['Hotel_Address'].values
        # validation
        if id_input not in train.id.values.tolist():
            sg.Popup("Invalid user Id")
        elif any(country_input in a for a in list_hotel) == False:
            sg.Popup("Invalid country/ city input")
        else:
            if norm_input != 'none':
                train = norm_test(norm_input,train)
            if filter_input != 'none':
                train = filter_test(filter_input, train)
            train = feature_test(feature_input, train, norm_input)

            hotelList = hotel_df(train,id_input,travel_input,room_input,travel_input)
            list1 = pd.Series(index=hotelList)
            for n in hotelList:
                pearson_corr = sim(train,id_input,n)
                print n
                list1[n] = pearson_corr
            top_sim = list1.sort_values(ascending=False).head(20)
            country = country_filter(top_sim,country_input)
            predict(country,country_input)

            test = pd.read_csv("1_filter_test.csv", na_values = ['no info', '.'], low_memory=False)
            train = pd.read_csv("1_filter_train.csv", na_values = ['no info', '.'], low_memory=False)
            final = pd.read_csv("rating_382.csv", na_values = ['no info', '.'], low_memory=False).head(20)
            test['Rating_adjusted']= test.Reviewer_Score-test.User_Average_Score
            train['Rating_adjusted']= train.Reviewer_Score-train.User_Average_Score

            full = pd.concat([test,train], axis=0, sort=False)

            full['Positive_Words_Percent'].fillna(0, inplace=True)
            full['Review_Date'] = pd.to_datetime(full['Review_Date']).astype('int64')

            if norm_input != 'none':
                full = norm_test(norm_input,full)
            full = feature_test(feature_input,full,norm_input)

            user_test = test[test['id']== id_input]
            full_test = full[full['id']== id_input]

            user_hotel = user_test.Hotel_Address.unique()
            test_final= full_test[full_test.Hotel_Address.isin(user_hotel)]
            test_final= test_final[test_final['Hotel_Address'].str.contains(country_input)]

            # calculate evaluation
            recall = get_recall(final,user_hotel)
            precision = get_precision(final,user_hotel)

            hotel_rec = final.Hotel_Address
            hotel_score = final.New_Rate

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
                hotel_10,line_width=500)

            sg.Popup("Hotel Recommendations:",
                'precision----------------------------------------------------------------------------------------',
                precision,
                'recall-------------------------------------------------------------------------------------------',
                round(recall,6),
                'f1-----------------------------------------------------------------------------------------------',
                round(get_f1(recall,precision),6),
                'mae----------------------------------------------------------------------------------------------',
                get_mae(final,test_final),
                'rmse---------------------------------------------------------------------------------------------',
                get_rmse(final,test_final),line_width=500)
