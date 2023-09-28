#  code referenced: https://www.toptal.com/machine-learning/nlp-tutorial-text-classification#:~:text=There%20are%20several%20NLP%20classification,progress%20notes%20at%20healthcare%20institutions.
import nltk, os, json
import numpy as np
import pandas as pd
import pickle
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


label_dir = {
        "small talk": "processed_data/conversation.json", 
        "hotel recommender" : "processed_data/hotel_location.csv",
        "hotel recommender" : "processed_data/recommend_hotel_prompt.json",
        "user review" : "processed_data/user_review.json",
        "destination list" : "processed_data/destination.json"
    }


def intent_classifier(input):

    data = [] 
    labels = [] 

    labels.append('hotel recommender')
    data.append('rating')
    for label in label_dir.keys():
        filepath = label_dir[label]
        if filepath.endswith(".csv"):    # small talk , name characteristics, hotel_location 
            csv_file = pd.read_csv(filepath)
            
            if filepath == "processed_data/hotel_location.csv":
                for row in csv_file.iterrows():
                    data.append(row[1]["city"].lower())
                    labels.append(label)

            elif filepath == "processed_data/trip_type.csv": 
                for col in csv_file.columns:
                    data.append(col.lower())
                    labels.append(label)
                    
        else : 
            json_file = json.load(open(filepath, "r"))
            for intent in json_file["intents"]:
                for pattern in intent["patterns"]:
                    data.append(pattern.lower())
                    labels.append(label)
        
        
       
    X_train , X_test , y_train , y_test = train_test_split ( data , labels , stratify = labels, test_size =0.25 , random_state =1)

    count_vect = CountVectorizer(stop_words = stopwords.words('english'))
    X_train_vect = count_vect.fit_transform( X_train )

    tfidf_transformer = TfidfTransformer ( use_idf = True , sublinear_tf = True ). fit(X_train_vect)
    X_train_tf = tfidf_transformer.transform ( X_train_vect)
    model = DecisionTreeClassifier(random_state =0).fit(X_train_tf, y_train )
  
    input = input.lower()
    new_data = [input]
    processed_newdata = count_vect.transform(new_data)
    processed_newdata = tfidf_transformer.transform(processed_newdata)
    intent = model.predict(processed_newdata)

   
    #Eval

    X_test_vect =  count_vect.transform(X_test)
    tfidf_transformer = TfidfTransformer ( use_idf = True , sublinear_tf = True ). fit(X_test_vect)
    X_test_tf = tfidf_transformer.transform(X_test_vect)
    predicted = model.predict(X_test_tf)

    # print(accuracy_score(y_test, predicted))
    

    return intent
