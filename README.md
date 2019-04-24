# UG-Recommender-System
<H2>Exploring the Effect of Rating and Reviews on Hotel Rankings to Create a Functioning Recommendation System</H2>
UG project - READ ME FILE

<H2>Dataset</H2>
<b>Hotel_Reviews.csv:</b>
Acknowledgement: Data was scraped from Booking.com, all data is originally owned by Booking.com
The hotel dataset is available from the following link:
https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe/activity

<H2>Prerequisites</H2>

<H2>Code files </H2>

<b>data_preprocess.py:</b>
Run this code first on the original Hote_Reviews.csv file to return the initial_filter.csv file necessary for the main program to run. It data pre-processes the dataset before recommendations can begin.
Requires: Hotel_Reviews.csv
Produces: initial_filter.csv

<b>dataset_longtail.py:</b>
This program is used to visualise the rating frequency by hotels and by users. The code has been altered from the original code by KevinLiao159 on github
Requires: inital_filter.csv

<b>recommender_system.command:</b>
This program contains the algorithm for the best quality algorithm. It allows users to select their id and choose options for their hotel search.
Note: If computer does not use Python 2.7, change the import library of pySimpleGUI27 to pySimpleGUI
Make sure the place the inital_filter.csv in the correct folder (The correct filepath is printed when the program is executed)
1. Run program
2. Select options (enter inputs)
3. Program will return a list of hotel recommendations
Requires: inital_filter.csv

<b>train_test.py:</b>
This program takes one train and test dataset to run the collaborative filtering test and returns the precision, recall, f1, mae and rmse results for that particular set of data
Must be run as a python file through terminal (only used for testing)
Note: If computer does not use Python 2.7, change the import library of pySimpleGUI27 to pySimpleGUI
Requires: train and test dataset: 1_filter_test.csv and 1_filter_train.csv

<b>avg_train_test.py:</b>
This program splits the dataset into into train and test 10 times and averages these results. Results will be different every time because the split is random for every run. (Is train_test.py, but run 10 times and averaged to return results)
Note: Due to the nature of this program, it does not support validation issues and will need to be re-run after every query
Must be run as a python file through terminal (only used for testing) every run should take around 20minutes to complete
Note: If computer does not use Python 2.7, change the import library of pySimpleGUI27 to pySimpleGUI
Requires: inital_filter.csv

<b>rank.py:</b>
This program allows comparison between two ranked lists and returns the kendall tau and spearman's rho values.
Requires: two lists of predicted rankings (must be edited in the code)

<H2>CSV Files</H2>
Sample test data:

<b>1_filter_train.csv.zip:</b>
Sample training set (70% of data)

<b>1_filter_test.csv.zip:</b>
Sample testing set (30% of data)

<H2>References</H2>
https://github.com/KevinLiao159/MyDataSciencePortfolio/blob/master/movie_recommender/movie_recommendation_using_KNN.ipynb



