import pandas as pd 
import numpy as np 






## get location -- do it separately and make it into a csv file
#hotel city in data frame
# hotel_names_coord = reviews[['Hotel_Name','Hotel_Address','lat','lng']].drop_duplicates().dropna()

# hotel_city= {}

# from geopy.geocoders import Nominatim

# # initialize Nominatim API
# geolocator = Nominatim(user_agent="geoapiExercises")

# # create a dictionary of hotels with their cities
# for name,lat,lng in zip(hotel_names_coord.Hotel_Name, hotel_names_coord.lat, hotel_names_coord.lng):
#     Latitude = str(lat)
#     Longitude = str(lng)
#     location = geolocator.reverse(Latitude+","+Longitude)
#     address = location.raw['address']
#     city = address.get('city', '')
#     hotel_city[name] = city
    


# def coord_to_city(lat,lng): 
#     Latitude = str(lat)
#     Longitude = str(lng)
#     location = geolocator.reverse(Latitude+","+Longitude)
#     address = location.raw['address']
#     city = address.get('city', '')
#     return city 

# hotel_names_coord['city'] = hotel_names_coord.apply(lambda row : coord_to_city(row['lat'],
#                      row['lng']),  axis = 1)


location = pd.read_csv("processed_data/hotel_location.csv")

location =location.dropna(subset = 'city')


location.to_csv(r'processed_data\hotel_location.csv')
