import sys
import csv
from pytube import YouTube

CSV_PATH = '../data-preprocessing/sample.csv'
VIDEO_SAVE_PATH = "Download-videos/"

# video stream filters
FILE_EXTENSION = 'mp4'
RESOLUTIONS = ['240p','360p','720p']   # resolution preference - first occurrence will be downloaded

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

    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    
    for row in csv_reader:
        line_count +=1
        if line_count == 1: continue    # ignore header line

        game_name = row[0] 
        video_links = row[3].strip('][').split(', ') # convert videolinks string to a list

        for j in range(len(video_links)): 
            try: 
                youtube_obj = YouTube(video_links[j])

                # filter streams with resolution list
                for i in range(len(RESOLUTIONS)):
                    filters = youtube_obj.streams.filter(progressive=True, file_extension=FILE_EXTENSION, res=RESOLUTIONS[i])
                    if(len(filters) > 0): break
                
                # download the first stream
                filters.first().download(output_path=VIDEO_SAVE_PATH, filename=game_name+'_video_'+str(j+1)+'.'+FILE_EXTENSION)
                
            except Exception as e:
                print(e)

print("End of the script")