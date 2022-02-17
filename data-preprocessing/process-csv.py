import csv
import os
import json
from youtubesearchpython import VideosSearch

DIR_PATH = '../data-files/'

sub_dirs = []
json_file_paths = []

for file in os.listdir(DIR_PATH):
    sub_dirs.append(file)

#  store json file names 
for sub_dir in sub_dirs:
    for file in  os.listdir(DIR_PATH+sub_dir+'/'):
        if file.endswith(".json"):
            json_file_paths.append(sub_dir+'/'+file)

# open csv file to write 
with open('processed_data.csv', 'w', newline='', encoding='utf-8') as csv_file:
    
    # add csv headers
    field_names = ['Game', 'Comfort_Class', 'Reviews', 'Real_Videos']
    csv_writer = csv.DictWriter(csv_file, fieldnames=field_names)
    csv_writer.writeheader()

    # load json file data
    for file in json_file_paths:
        with open(DIR_PATH+file, 'r') as json_file:
            
            data = json.load(json_file)
            
            game_name = data['display_name']
            comfort_class = data['comfort_rating'] if ('comfort_rating' in data) else ''
            
            review_edges = data['firstQualityRatings']['edges'] if ('firstQualityRatings' in data and 'edges' in data['firstQualityRatings']) else []
            reviews = []
            
            for i in range(len(review_edges)):
                reviews.append(review_edges[i]['node']['reviewDescription']) 

            # video search    
            videosSearch = VideosSearch(game_name + ' VR', limit = 2)
            vr_gameplay_video_links=[videosSearch.result()['result'][i]['link'] for i in range(len(videosSearch.result()['result']))] 

            csv_writer.writerow({'Game': game_name, 'Comfort_Class': comfort_class, 'Reviews': reviews, 'Real_Videos': vr_gameplay_video_links}) 

