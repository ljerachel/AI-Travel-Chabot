import pandas as pd
import numpy as np 
import nltk, os, json
import pickle
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


#filter out the hotels by name for k cluster to match additional comments of user to pos/neg reviews 
def filter_by_user_pref(userpref):  
    location = userpref['loc_query']
    rating = userpref['rating_query']
    # room_type = userpref['room_type']
    trip_type = userpref['trip_type']

    # filter by location and rating
    filepath = "processed_data/hotel_location.csv"
    hotel_loc = pd.read_csv(filepath)
    hotel_names = hotel_loc.loc[(hotel_loc["city"] == location) & (hotel_loc["Average_Score"] >= rating), "Hotel_Name"]
   

    trip = pd.read_csv("processed_data/trip_type.csv")

    filtered_trip = pd.merge(hotel_names,trip,on='Hotel_Name')  
    filtered_trip = filtered_trip.loc[(filtered_trip[trip_type]== 1 ), "Hotel_Name"]

    # merge filtered hotel names with reviews 
    pos_reviews = pd.read_csv("processed_data/positive_reviews.csv")
    filtered_trip_reviews = pd.merge(filtered_trip,pos_reviews, on='Hotel_Name') 
    filtered_trip_reviews.to_csv(r'processed_data/filtered_trip_reviews.csv')

    # return hotel_names

#code referenced: https://www.toptal.com/machine-learning/nlp-tutorial-text-classification#:~:text=There%20are%20several%20NLP%20classification,progress%20notes%20at%20healthcare%20institutions.
def classify_hotels(user_input):
    filepath = 'processed_data/filtered_trip_reviews.csv'
    csv_file = pd.read_csv(filepath)
    hotels = [] 
    reviews = [] 
    for row in csv_file.iterrows():
        reviews.append(row[1]["reviews"])
        hotels.append(row[1]["Hotel_Name"])
       
    length = len(csv_file['Hotel_Name'].unique())
    X_train , X_test , y_train , y_test = train_test_split (reviews  , hotels , stratify = hotels, test_size =0.25 , random_state =1)

    count_vect = CountVectorizer(stop_words = stopwords . words ('english'))
    X_train_counts = count_vect.fit_transform( X_train )

    tfidf_transformer = TfidfTransformer ( use_idf = True , sublinear_tf = True ). fit (
    X_train_counts )
    X_train_tf = tfidf_transformer.transform ( X_train_counts )

    model = SVC(C = 1 , gamma= 0.001).fit(X_train_tf, y_train)
    new_data = [user_input]
    processed_newdata = count_vect.transform(new_data)
    processed_newdata = tfidf_transformer.transform(processed_newdata)
    label = model.predict(processed_newdata)
    

    X_test_vect =  count_vect.transform(X_test)
    tfidf_transformer = TfidfTransformer ( use_idf = True , sublinear_tf = True ). fit(X_test_vect)
    X_test_tf = tfidf_transformer.transform(X_test_vect)
    predicted = model.predict(X_test_tf)

    # print(accuracy_score(y_test, predicted))

    return label

