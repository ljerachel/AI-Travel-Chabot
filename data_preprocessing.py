import pandas as pd 
import numpy as np 
from geopy.geocoders import Nominatim


reviews =pd.read_csv(r'data\Hotel_Reviews.csv', nrows = 100000) 
reviews.dropna()
reviews.drop(['Review_Date', 'Additional_Number_of_Scoring', 'Review_Total_Negative_Word_Counts','Review_Total_Positive_Word_Counts','Total_Number_of_Reviews_Reviewer_Has_Given', 'days_since_review','Total_Number_of_Reviews'], axis = 1, inplace = True)

# remove unreliable reviewers by thresholding reviewer score 
reviews = reviews.loc[reviews["Reviewer_Score"] > 5]

# reviews['Tags'] = reviews.Tags.apply(lambda x: x[1:-1].split(','))
# sub_binary = pd.get_dummies(reviews.Tags.explode()).sum(level=0)
# sub_binary = sub_binary.rename(columns=lambda x: x.strip("'").replace("'",""))
# hotel_names = reviews['Hotel_Name']
# hotel_names = pd.DataFrame(hotel_names)

# hotel_char = hotel_names.join(sub_binary)
# hotel_char.columns = hotel_char.columns.str.strip()

## additional comments (k means clustering label = hotel name)
pos_reviews = reviews[['Hotel_Name', 'Positive_Review']]
neg_reviews = reviews[['Hotel_Name', 'Negative_Review']]

# # remove No negative and No Positive in data set 
pos_reviews = pos_reviews[pos_reviews.Positive_Review != 'No Positive']

neg_reviews = neg_reviews[neg_reviews.Negative_Review != 'No Negative']

pos_reviews = pos_reviews.rename(columns = {'Positive_Review': 'reviews'})
neg_reviews = neg_reviews.rename(columns = {'Negative_Review': 'reviews'})

# # separate tags into type of trip and room type
# trip_type = hotel_char[["Hotel_Name", "Business trip", "Couple", "Family with older children", "Family with young children", "Group", "Leisure trip", "Solo traveler", "Travelers with friends", "With a pet"]]

# # get location -- do it separately and make it into a csv file
# hotel_names_coord = reviews[['Hotel_Name','Hotel_Address','lat','lng','Average_Score']].drop_duplicates().dropna()


# initialize Nominatim API
# geolocator = Nominatim(user_agent="geoapiExercises")


# def coord_to_city(lat,lng): 
#     Latitude = str(lat)
#     Longitude = str(lng)
#     location = geolocator.reverse(Latitude+","+Longitude)
#     address = location.raw['address']
#     city = address.get('city', '')
#     return city 

# hotel_names_coord['city'] = hotel_names_coord.apply(lambda row : coord_to_city(row['lat'],
#                      row['lng']),  axis = 1)
                     
# hotel_names_coord =hotel_names_coord.dropna(subset = 'city')


# save in file
neg_reviews.to_csv(r'processed_data\negative_reviews.csv')
pos_reviews.to_csv(r'processed_data\positive_reviews.csv')
# trip_type.to_csv(r'processed_data\trip_type.csv')
# hotel_names_coord.to_csv(r'processed_data\hotel_location.csv')







