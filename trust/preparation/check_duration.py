# by Pavlo Bazilinskyy <pavlo.bazilinskyy@gmail.com>

import os
import pandas as pd
import trust as tr
from ffprobe import FFProbe


# read mapping csv
df = pd.read_csv('process_videos_info.csv')

for index, row in df.iterrows():
    # adapt path for python
    folder = tr.common.get_configs('path_stimuli')
    folder = folder.replace('\\\u0020', ' ')
    in_file = os.path.join(folder, row['out'])
    # https://stackoverflow.com/a/3844467/46687
    print(row['out'], int(float(FFProbe(in_file).video[0].duration) * 1000))
