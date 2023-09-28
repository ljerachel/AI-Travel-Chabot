#code referenced: https://www.toptal.com/machine-learning/nlp-tutorial-text-classification#:~:text=There%20are%20several%20NLP%20classification,progress%20notes%20at%20healthcare%20institutions.
import nltk, os, json
import numpy as np
import pandas as pd
import pickle
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier


def review_classifier(input):
    label_dir = {
            "positive": "processed_data/positive_reviews.csv", 
            "negative" : "processed_data/negative_reviews.csv"
        }

    data = [] 
    labels = [] 

    for label in label_dir.keys():
        filepath = label_dir[label]
        if filepath.endswith(".csv"):    # small talk , name characteristics, hotel_location 
            csv_file = pd.read_csv(filepath)
            for row in csv_file.iterrows():
                data.append(row[1]["reviews"])
                labels.append(label)
                
       
    X_train , X_test , y_train , y_test = train_test_split ( data , labels , stratify = labels, test_size =0.25 , random_state =1)

    count_vect = CountVectorizer(stop_words = stopwords . words ('english'))
    X_train_counts = count_vect.fit_transform( X_train )

    tfidf_transformer = TfidfTransformer ( use_idf = True , sublinear_tf = True). fit (X_train_counts)
    X_train_tf = tfidf_transformer.transform ( X_train_counts )
    model = LogisticRegression(max_iter = 200).fit(X_train_tf, y_train)
   
    # load the model from disk
    filename = 'MLmodels/classifyreviews.sav'
    model = pickle.load(open(filename, 'rb'))
        
    input = input.lower()
    new_data = [input]
    processed_newdata = count_vect.transform(new_data)
    processed_newdata = tfidf_transformer.transform(processed_newdata)
    review_class = model.predict(processed_newdata)
    
    # save ML model
    # pickle.dump(model, open(filename, 'wb'))
    
    X_test_vect =  count_vect.transform(X_test)
    tfidf_transformer = TfidfTransformer ( use_idf = True , sublinear_tf = True ). fit(X_test_vect)
    X_test_tf = tfidf_transformer.transform(X_test_vect)
    predicted = model.predict(X_test_tf)
   
    # print(accuracy_score(y_test, predicted))
    
    return review_class 

