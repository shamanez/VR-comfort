from pytube import YouTube

SAVE_PATH = "Download-videos/"
# FILE_NAME = "download.mp4"

# video stream filters
FILE_EXTENSION = 'mp4'
RESOLUTIONS = ['240p','360p','720p']   # resolution preference order - first occurrence will be downloaded

youtube_video_urls = ['https://www.youtube.com/watch?v=DkU9WFj8sYo']

for video_link in youtube_video_urls: 
    try: 
        yt_obj = YouTube(video_link)

        # filter streams with resolution list
        for i in range(len(RESOLUTIONS)):
            filters = yt_obj.streams.filter(progressive=True, file_extension=FILE_EXTENSION, res=RESOLUTIONS[i])
            if(len(filters) > 0): break

         # download the first stream
        filters.first().download(output_path=SAVE_PATH)
        
    except Exception as e:
        print(e)

print("End of the script")