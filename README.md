# UG---Recommender-System
UG project - READ ME FILE

Files:

<b>initial_filter.py:</b>
This is run first - it is the data pre-processing stage and is run on the original dataset taken from Kaggle. 
Requires: Hotel_Reviews.csv
Produces: initial_filter.csv

<b>dataset_longtail.py:</b>
This program is used to visualise the rating frequency by hotels and by users. The code has been altered from the original code by KevinLiao159 on github
Requires: inital_filter.csv

<b>recommender_system.command:</b>
This program contains the algorithm for the best quality algorithm. It allows users to select their id and choose options for their hotel search.
Make sure the place the inital_filter.csv in the correct folder.
1. Run program
2. Select options (enter inputs)
3. Program will return a list of hotel recommendations
Requires: inital_filter.csv

<b>train_test.py:</b>
This program takes one train and test dataset to run the collaborative filtering test and returns the precision, recall, f1, mae and rmse results for that particular set of data
Requires: train and test dataset: 1_filter_test.csv and 1_filter_train.csv

<b>avg_train_test.py:</b>
This program splits the dataset into into train and test 10 times and averages these results. Results will be different every time because the split is random for every run. (Is train_test.py, but run 10 times and averaged to return results)
Note: Due to the nature of this program, it does not support validation issues and will need to be re-run after every query
Requires: inital_filter.csv

<b>rank.py:</b>
This program allows comparison between two ranked lists and returns the kendall tau and spearman's rho values.
Requires: two lists of predicted rankings

<b>Hotel_Reviews.csv:</b>
The original dataset downloaded from Kaggle. Acknowledgement: Data was scraped from Booking.com, all data is originally owned by Booking.com
This dataset is not located in this repository, please download it from the following link:
https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe/activity

<b>initial_filter.csv:</b>
The filtered dataset, it is created by running initial_filter.py

<b>1_filter_train.csv:</b>
Sample training set (70% of data)

<b>1_filter_test.csv:</b>
Sample testing set (30% of data)



