'''
This script will try the install and import of dlc,torch.

'''


#%%
# import deeplabcut
# import torch
import ffmpeg


start_time = '01:47:00' # Start time for trimming (HH:MM:SS)
end_time = '01:57:00' # End time for trimming (HH:MM:SS)

(
	ffmpeg.input(r"D:\ZR\dlc_project\Raw_Videos\20250118_SRS4.mp4",ss=start_time, to=end_time)
	.output(r"D:\ZR\dlc_project\cropped\cutted.mp4")
	.run()

)

#%% crop video


# Apply cropping: crop=width:height:x:y
try:
    input_video = ffmpeg.input(r'D:\ZR\dlc_project\cropped\cutted.mp4')
    cropped = input_video.filter('crop', w=900, h=540, x=960, y=540)
    output = ffmpeg.output(cropped, r'D:\ZR\dlc_project\single_animal\output.mp4')
    # Run and capture stderr
    ffmpeg.run(output, overwrite_output=True, capture_stdout=True, capture_stderr=True)
except ffmpeg.Error as e:
    print("FFmpeg stderr:", e.stderr.decode())