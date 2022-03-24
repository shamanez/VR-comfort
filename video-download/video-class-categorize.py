import os
import csv
import sys
import re
import shutil
from collections import defaultdict

CSV_PATH = "../data-preprocessing/processed_data.csv"
VIDEOS_DIR = "video-subclips"
SAVE_DIR = "../VR-dataset-final"
SAVE_TRAIN_DIR = "../VR-dataset-final/train"
SAVE_VALIDATE_DIR = "../VR-dataset-final/validate"

class_gamevideofiles_dict = defaultdict(list)
gamename_subclipfiles_dict = defaultdict(list)

train_videos = []
validate_videos = []

train_data_percentage = 0.8

# making final dataset directories 
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

if not os.path.exists(SAVE_TRAIN_DIR):
    os.makedirs(SAVE_TRAIN_DIR)
    
if not os.path.exists(SAVE_VALIDATE_DIR):
    os.makedirs(SAVE_VALIDATE_DIR)


# map subclip-files with game_name
for file in  os.listdir(VIDEOS_DIR):
    if file.endswith(".mp4"):
        game_name = re.split("_\d$", file.split('.mp4')[0].split('_clip_')[0])[0]
        gamename_subclipfiles_dict[game_name].append(file)


# for csv field limit error
maxInt = sys.maxsize
while True:
    # decrease the maxInt value by factor 10 
    # as long as the OverflowError occurs.
    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)

# open csv file to read  
with open(CSV_PATH, 'r', encoding='utf-8') as csv_file:
    csv_reader = csv.DictReader(csv_file, delimiter=',')

    for row in csv_reader:
        game_name=row['Game']
        comfort_class= row['Comfort_Class'].strip(' ')

        if game_name in gamename_subclipfiles_dict.keys():
            class_gamevideofiles_dict[comfort_class].append(gamename_subclipfiles_dict[game_name])


for comfort_class in class_gamevideofiles_dict:

    # make directory for comfort-class inside train
    if not os.path.exists(os.path.join(SAVE_TRAIN_DIR, comfort_class)):
        os.makedirs(os.path.join(SAVE_TRAIN_DIR, comfort_class))

    # make directory for comfort-class inside validate
    if not os.path.exists(os.path.join(SAVE_VALIDATE_DIR, comfort_class)):
        os.makedirs(os.path.join(SAVE_VALIDATE_DIR, comfort_class))
    
    num_of_train_videos = round(len(class_gamevideofiles_dict[comfort_class]) * (train_data_percentage)) 

    # taining video-clips
    for subclip_list in class_gamevideofiles_dict[comfort_class][:num_of_train_videos]:
        for subclip in subclip_list:
            source =  os.path.join(VIDEOS_DIR, subclip)
            destination = os.path.join(SAVE_TRAIN_DIR, comfort_class, subclip)
            shutil.move(source, destination)

    # validate video-clips
    for subclip_list in class_gamevideofiles_dict[comfort_class][num_of_train_videos:]:
        for subclip in subclip_list:
            source =  os.path.join(VIDEOS_DIR, subclip)
            destination = os.path.join(SAVE_VALIDATE_DIR, comfort_class, subclip)
            shutil.move(source, destination)

