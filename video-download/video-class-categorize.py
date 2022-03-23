import os
import csv
import sys
import re
import math
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

        # extend subclip-files-list for the comfort class
        class_gamevideofiles_dict[comfort_class].extend(gamename_subclipfiles_dict[game_name])



for comfort_class in class_gamevideofiles_dict:

    # make directory for comfort-class inside train
    if not os.path.exists(os.path.join(SAVE_TRAIN_DIR, comfort_class)):
        os.makedirs(os.path.join(SAVE_TRAIN_DIR, comfort_class))

    # make directory for comfort-class inside validate
    if not os.path.exists(os.path.join(SAVE_VALIDATE_DIR, comfort_class)):
        os.makedirs(os.path.join(SAVE_VALIDATE_DIR, comfort_class))
    
    # 80% videos from every comfortclass will go into taining, 
    # 20% videos into validation
    train_videos = math.ceil(len(class_gamevideofiles_dict[comfort_class]) * (0.8)) 

    # taining video-clips
    for videofile in class_gamevideofiles_dict[comfort_class][:train_videos]:
        source =  os.path.join(VIDEOS_DIR, videofile)
        destination = os.path.join(SAVE_TRAIN_DIR, comfort_class, videofile)
        shutil.move(source, destination)

    # validate video-clips
    for videofile in class_gamevideofiles_dict[comfort_class][train_videos:]:
        source =  os.path.join(VIDEOS_DIR, videofile)
        destination = os.path.join(SAVE_VALIDATE_DIR, comfort_class, videofile)
        shutil.move(source, destination)

