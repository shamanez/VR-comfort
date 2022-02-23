import csv
import os
import json
from youtubesearchpython import VideosSearch

DIR_PATH = '/home/gsir059/Documents/game-reviews/'

sub_dirs = []
json_file_paths = []
gear_type=[]
name_list=[]

for file in os.listdir(DIR_PATH):
    sub_dirs.append(file)

#  store json file names 
for sub_dir in sub_dirs:
    for file in  os.listdir(DIR_PATH+sub_dir+'/'):
        if file.endswith(".json"):
            json_file_paths.append(sub_dir+'/'+file)
            gear_type.append(sub_dir)

print('Number of games :',len(json_file_paths))
# open csv file to write 
with open('processed_data.csv', 'w', newline='', encoding='utf-8') as csv_file:
    
    # add csv headers
    field_names = ['Game', 'Comfort_Class', 'Reviews','Review_Scores', 'Real_Videos','Gear']
    csv_writer = csv.DictWriter(csv_file, fieldnames=field_names)
    csv_writer.writeheader()

    # load json file data
    for file,gear_name in zip(json_file_paths,gear_type):
        with open(DIR_PATH+file, 'r') as json_file:
            
            data = json.load(json_file)
            
            game_name = data['display_name']
            comfort_class = data['comfort_rating'] if ('comfort_rating' in data) else ''
            if len(comfort_class) <1:
                continue
            if game_name in name_list:
                continue
            name_list.append(game_name)
            
        
            
            review_edges = data['firstQualityRatings']['edges'] if ('firstQualityRatings' in data and 'edges' in data['firstQualityRatings']) else []
            reviews = []
            scores=[]
            
            for i in range(len(review_edges)):
                reviews.append(review_edges[i]['node']['reviewDescription']) 
                scores.append(review_edges[i]['node']['score'])
            
   
           
            # video search    
            videosSearch = VideosSearch(game_name +' '+ ' VR gameplay',limit = 2)
            vr_gameplay_video_links=[videosSearch.result()['result'][i]['link'] for i in range(len(videosSearch.result()['result']))] 

            csv_writer.writerow({'Game': game_name, 'Comfort_Class': comfort_class, 'Reviews': reviews,'Review_Scores': scores, 'Real_Videos': vr_gameplay_video_links,'Gear':gear_name}) 

