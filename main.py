
import os, json, random
import pandas as pd
from csv import DictWriter
from intent_classifier import intent_classifier
from  similarity import cosine_similarity
from hotel_recommender import filter_by_user_pref, classify_hotels
from classify_reviews import review_classifier
import re

def get_input(user_names, bye_bot, user_input, userpref):
    max_sim = float('-inf')
    response = ""
    input = [user_input]  ## check this one 
    bot = []
    topic = []

    # if chatbot cannot match intent(recommend)
    chatbot_apology = "My apologies, could you rephrase that so I can assist you better?"

    # classify intent
    user_intent = intent_classifier(user_input)
    name_exists = False
    if user_intent == 'small talk':
        filepath = r"processed_data/conversation.json"
        json_file = json.load(open(filepath, "r"))
        for intent in json_file["intents"]:
            for pattern in intent["patterns"]:
                sim = cosine_similarity(user_input, pattern)
                if sim > max_sim:
                    max_sim = sim
                    response = random.choice(intent["responses"])
                    tag = intent['tag']

                    if max_sim < 0.2:
                        response = chatbot_apology

        if tag == 'NameResponse':
            name = get_user_name(user_input)
            if name in user_names:
                name_exists = True
                response = "Welcome back, {}! How can I help you today?".format(name)

            else:
                response = response.replace('<HUMAN>', name)
            userpref['username'] = name

        elif tag == 'CurrentUserNameQuery':
            user_name = list(filter(None, user_name))
            if userpref['username'] != " ":
                response = response.replace('<HUMAN>', userpref['username'])
            else:
                response = "I'm sorry, I dont think we've met yet!"

        elif tag == "GoodBye":
            bye_bot = True 
        
        print("Bot: " + response)
        bot.append(response)

    elif user_intent == 'hotel recommender':
        response = "Please tell me the location you wish to travel to"
        print("Bot: " + response)
        user_input = getnewinput()
        filepath = r"processed_data/hotel_location.csv" 
        csv_file = pd.read_csv(filepath)
        for row in csv_file.iterrows():
            location = str(row[1]['city']) 
            sim = cosine_similarity(user_input, location)
            if sim >= max_sim:
                max_sim = sim
                chosen_location  = location 

        if max_sim < 0.3:
            response = "Sorry, I cannot assist you with your accomodation in this location."
            print("Bot: " + response)  

        else :
            userpref['loc_query']=chosen_location# save location query
            response = "Searching for hotels in {}. Please tell me the your desired hotel ratings :D".format(chosen_location)
            print("Bot: " + response)
            bot.append(response)    
            user_input = getnewinput()
            rating = get_user_rating(user_input)  
            userpref['rating_query'] = rating
            response = "Aye aye! Hotels with a rating of {} are now under my radar. Can you tell me about the trip you're planning".format(rating)
            print("Bot: " + response)
            bot.append(response)       
            user_input = getnewinput()
            filepath = r"processed_data/trip_type.csv" 
            csv_file = pd.read_csv(filepath)
            for column in csv_file.columns:
                trip_type  = column
                sim = cosine_similarity(user_input, trip_type)
                if sim >= max_sim:
                        max_sim = sim
                        if max_sim < 0.3:
                            response = chatbot_apology

                        else :
                            userpref['trip_type']=trip_type# save trip type query
                            response = "Nice! Now would you describe your ideal stay for the trip ? "
                            filtered_trip = filter_by_user_pref(userpref)  # function to filter data frame to get the final hotel names 
                            print("Bot: " + response)
                            bot.append(response)
                            user_input = getnewinput()
                            hotel = classify_hotels(user_input)
                    
                            response  = "I would strongly recommend {} :D".format(hotel)


                            userpref['recommended_hotel'] = hotel
                            print("Bot: " + response)
                                    

    
    elif user_intent == 'user review':
        add_hotel_review = {"hotel_name": '', "positive_review":'', "negative_review":''}
        filepath = r"processed_data/user_review.json"
        json_file = json.load(open(filepath, "r"))
        for intent in json_file["intents"]:
            for pattern in intent["patterns"]:
                sim = cosine_similarity(user_input, pattern)
                if sim > max_sim:
                    max_sim = sim
                    response = random.choice(intent["responses"])
                    tag = intent['tag']

                    if max_sim < 0.2:
                        response = chatbot_apology
        if tag == 'UserReview':
            bot.append(response)
            print("Bot: " + response)
            hotel_name_query = getnewinput()
            add_hotel_review['hotel_name'] = hotel_name_query
            response = "Please tell me your thoughts on this place"
            print("Bot: " + response)
            review = getnewinput()
            review_class = review_classifier(review)
            if review_class == 'positive':
                response = "Thank you for the positive review"
                add_hotel_review['positive_review'] = review
                add_hotel_review['negative_review'] = ''
                
            elif review_class == 'negative':
                response = "We're sorry to hear that, we will be sure to inform {} about your experience".format(hotel_name_query)
                add_hotel_review['positive_review'] = ''
                add_hotel_review['negative_review'] = review

            print("Bot: "+ response)
            # put in csv file
            field_names = ['hotel_name', 'positive_review', 'negative_review']
            with open('SaveData/user_reviews.csv', 'a') as csvfile:
                dictwriter_object = DictWriter(csvfile, fieldnames=field_names)
                dictwriter_object.writerow(add_hotel_review)
                csvfile.close()

    elif user_intent =='destination list':
        filepath = r"processed_data/destination.json"
        json_file = json.load(open(filepath, "r"))
        for intent in json_file["intents"]:
            for pattern in intent["patterns"]:
                sim = cosine_similarity(user_input, pattern)
                if sim > max_sim:
                    max_sim = sim
                    response = random.choice(intent["responses"])
                    tag = intent['tag']

                    if max_sim < 0.2:
                        response = random.choice(chatbot_apology)
        if  tag == 'TravelList':
            # print("Bot: " + response )
            dest = get_dest_add(user_input)
            userpref['travel_list'].append(dest)
            response = dest +' has been added to your travel list >.<'
            print("Bot: "+ response )

        elif tag == 'deleteTrip':
            dest_delete = get_dest_delete(user_input)
            userpref['travel_list'].remove(dest_delete)
            response = response.replace('<DESTINATION>',dest_delete)
            print("Bot: " + response )

        elif tag == 'fetchTrip':
            travelList = '\n'.join(userpref['travel_list'])
            response = response.replace('<TRAVEL LIST>', travelList)
            print("Bot: " + response )

    return  bye_bot, userpref, name_exists
    
def get_dest_delete(user_input):
    texts = ["delete (.*) from my list", "remove (.*) from my list", "get rid of (.*) from my list"]
    reply = ''

    for text in texts:
        text = text.lower()
        regex = re.search(text, user_input)
        if regex:
            reply = re.findall(text, user_input)
    reply = str(reply)
    reply = reply.replace('[', '')
    reply = reply.replace(']', '')
    reply = reply.replace("'", '')
    return reply

def get_dest_add(user_input):
    texts = ["add (.*) to my travel list", "put (.*) in my travel list"]
    reply = ''

    for text in texts:
        text = text.lower()
        regex = re.search(text, user_input)
        if regex:
            reply = re.findall(text, user_input)
    reply = str(reply)
    reply = reply.replace('[', '')
    reply = reply.replace(']', '')
    reply = reply.replace("'", '')
    return reply


def getnewinput():
    user_input = input("User: ")
    return user_input


def get_user_name(user_input):
    texts = ["My name is (.*)", "I am (.*)", "It is (.*)", "call me (.*)", "This is (.*)", "I'm (.*)"]
    reply = ''

    for text in texts:
        text = text.lower()
        user_input = user_input.lower()
        regex = re.search(text, user_input)
        if regex:
            reply = re.findall(text, user_input)

    reply = str(reply)
    reply = reply.replace('[', '')
    reply = reply.replace(']', '')
    reply = reply.replace("'", '')
    reply = reply.capitalize()
    return reply

def get_user_rating(user_input):
    rating = re.findall(r'\d+', user_input)
    return float(rating[0])


users  = pd.read_csv('SaveData/user_identities.csv')
user_names = list(users['username'])

print("Bot: Hi! I'm Travelo : D Your friendly travel chatbot. ")

userchat = {'username':'', 'loc_query' : '', 'rating_query': 0 , 'trip_type': '','recommended_hotel': '', 'travel_list': []}
bye_bot = False
while not bye_bot :
    user_text = input("User: ")
    bye_bot,userchat,name_exists = get_input(user_names, bye_bot, user_text, userchat)

    if name_exists is True:
        current_user = users.loc[users['username'] == userchat['username']]
        userchat =  current_user.to_dict('series')  ### read line into dict

field_names = ['username', 'loc_query', 'rating_query','trip_type','recommended_hotel', 'travel_list']
with open('SaveData/user_identities.csv', 'a') as csvfile:
    dictwriter_object = DictWriter(csvfile, fieldnames=field_names)
    dictwriter_object.writerow(userchat)
    csvfile.close()
quit() 
    
    
    
        
