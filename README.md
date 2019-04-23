# UG---Recommender-System
UG project

READ ME FILE

Files:

initial_filter.py
This is run first - it is the data pre-processing stage and is run on the original dataset taken from Kaggle. 
Requires: Hotel_Reviews.csv
Produces: initial_filter.csv

dataset_longtail.py
This program is used to visualise the rating frequency by hotels and by users. The code has been altered from the original code by KevinLiao159 on github
Requires: inital_filter.csv

recommender_system.command
This program contains the algorithm for the best quality algorithm. It allows users to select their id and choose options for their hotel search.
Make sure the place the inital_filter.csv in the correct folder.
1. Run program
2. Select options (enter inputs)
3. Program will return a list of hotel recommendations
Requires: inital_filter.csv

train_test.py
This program takes one train and test dataset to run the collaborative filtering test and returns the precision, recall, f1, mae and rmse results for that particular set of data
Requires: train and test dataset: 1_filter_test.csv and 1_filter_train.csv

avg_train_test.py
This program splits the dataset into into train and test 10 times and averages these results. Results will be different every time because the split is random for every run. (Is train_test.py, but run 10 times and averaged to return results)
Note: Due to the nature of this program, it does not support validation issues and will need to be re-run after every query
Requires: inital_filter.csv

rank.py
This program allows comparison between two ranked lists and returns the kendall tau and spearman's rho values.
Requires: two lists of predicted rankings
