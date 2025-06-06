'''
This script will use ffmpeg to cut videos. This demo is useful for long video or other test situation.
'''

#%% 
import ffmpeg

start_time = '00:47:00' # Start time for trimming (HH:MM:SS)
end_time = '00:49:00' # End time for trimming (HH:MM:SS)

(
	ffmpeg.input(r"D:\_DataTemp\241213_Awake_Video\full_video.mp4",ss=start_time, to=end_time)
	.output(r"D:\_DataTemp\241213_Awake_Video\cutted.mp4")
	.run()

)

