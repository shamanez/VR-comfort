import sys
import csv
import os
import re
import matplotlib.pyplot as plt

CSV_PATH = "../data-preprocessing/processed_data.csv"
VIDEOS_DIR = "../video-download/video-subclips"

video_count_dict={}
class_video_count_dict={}

# count number of video clips for a game_name
for file in  os.listdir(VIDEOS_DIR):
    if file.endswith(".mp4"):
        game_name = re.split("_\d$", file.split('.mp4')[0].split('_clip_')[0])[0]
        video_count_dict[game_name]=1 if not game_name in video_count_dict else video_count_dict[game_name]+1 


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
        
        if game_name in video_count_dict:
            if comfort_class in class_video_count_dict:
                class_video_count_dict[comfort_class] += video_count_dict[game_name]
            else:
                class_video_count_dict[comfort_class] = video_count_dict[game_name]
        
        elif not comfort_class in class_video_count_dict:
            class_video_count_dict[comfort_class] = 0


# plot the class-video-count dictionary
plt.figure(figsize=(12,6))
plt.bar(*zip(*class_video_count_dict.items()), width = 0.3)

# value labels
for index, value in enumerate(class_video_count_dict.values()):
    plt.text(y=value+0.1, x=index, s=str(value))

plt.xlabel("Comfort Class")
plt.ylabel("Number of video clips")
plt.title("Number of video clips for comfort classes")
plt.show()