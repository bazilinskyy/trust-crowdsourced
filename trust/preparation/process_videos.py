# by Pavlo Bazilinskyy <pavlo.bazilinskyy@gmail.com>

import os
import pandas as pd
import trust as tr

print(os.path.join(tr.common.get_configs('path_stimuli'), "out.mp4"))

# Read mapping csv
df = pd.read_csv('process_videos_info.csv')
# black file to be added in the beginning of the stimuli
black_file = 'black_video.mp4'
# Go over all stimuli
for index, row in df.iterrows():
    in_file = os.path.join(tr.common.get_configs('path_source'), row['in'])
    out_file = os.path.join(tr.common.get_configs('path_stimuli'),
                            'noblack_' + row['out'])
    merged_file = os.path.join(tr.common.get_configs('path_stimuli'),
                               row['out'])
    # Using FFmpeg command to process video and compress audio
    os.system("ffmpeg -r 60" +
              " -ss " + str(row['start']) +
              " -to " + str(row['end']) +
              " -i " + in_file +
              " -c:v libx264 -profile:v baseline -level 3.0 -pix_fmt yuv420p" +
              " -c:a aac -b:a 32k" +
              " -vf scale=1280:720" +
              " -movflags faststart" +
              " -crf 23 " +
              out_file)
    # Using FFmpeg command to add 1 sec of black in the beginning
    # based on https://stackoverflow.com/a/22688066/46687
    os.system("ffmpeg -i " + black_file +
              " -i " + out_file +
              " -filter_complex \"[0:v:0] [0:a:0] [1:v:0] [1:a:0]" +
              " concat=n=2:v=1:a=1 [v] [a]\" -map \"[v]\" -map \"[a]\" -y " +
              merged_file)
