import ffmpeg
import os
import math
import random

VIDEOS_DIR = os.getcwd() + "/youtube-dataset"
SAVE_DIR = os.getcwd() + "/video-subclips"

NUM_OF_SUB_CLIPS = 5
SUB_CLIP_DURATION_SECS = 30 

mp4_files = {}

# store all mp4 file paths 
for file in  os.listdir(VIDEOS_DIR):
    if file.endswith(".mp4"):
        mp4_files[file] = (VIDEOS_DIR+ "\\" + file)  
        # mp4_files[file] = (VIDEOS_DIR+ "/" + file) # linux

# change directory for save clips
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

os.chdir(SAVE_DIR) 


for video_file in mp4_files:
    
    video_name = video_file.split('.')[0]
    video_path = mp4_files[video_file]
    video_length = math.floor(float(ffmpeg.probe(video_path)['format']['duration']))
    
    if (video_length < SUB_CLIP_DURATION_SECS):
        os.system('copy "' + video_path + '" "' + SAVE_DIR+'/'+ video_file +'"') # for linux use 'cp' instead 'copy'
        continue

    # check whether the required number of clips with required duration can be obtained
    # if not possible - get maximum possible number  
    max_sub_clips = NUM_OF_SUB_CLIPS  if((video_length // SUB_CLIP_DURATION_SECS) > NUM_OF_SUB_CLIPS) else (video_length // SUB_CLIP_DURATION_SECS)
       
   
    for i in range(max_sub_clips):
        
        random_start = random.randint(int(video_length*(i+1)/(max_sub_clips+2)), int(video_length*(i+2)/(max_sub_clips+2)))
     
        # video+audio trim
        stream = ffmpeg.input(video_path)
        audio = stream.audio.filter('atrim', start=random_start, duration=SUB_CLIP_DURATION_SECS).filter('asetpts', 'PTS-STARTPTS')
        video = stream.trim(start=random_start, duration=SUB_CLIP_DURATION_SECS).filter('setpts', 'PTS-STARTPTS')

        joined = ffmpeg.concat(video, audio, v=1, a=1).node
        output = ffmpeg.output(joined[0],joined[1], video_name+'_clip_'+str(i+1)+'.mp4')

        ffmpeg.run(output, overwrite_output=True)

