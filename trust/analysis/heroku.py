# by Pavlo Bazilinskyy <pavlo.bazilinskyy@gmail.com>
import json
import os
import pandas as pd
from statsmodels.stats.anova import AnovaRM
import numpy as np
import re
from tqdm import tqdm
from statistics import mean
import warnings
from shapely.geometry.polygon import Polygon
from shapely.geometry import Point
import trust as tr
import scipy.stats as stats
from scipy.stats import ks_1samp, norm
from scipy.stats import shapiro, levene

# warning about partial assignment
pd.options.mode.chained_assignment = None  # default='warn'

logger = tr.CustomLogger(__name__)  # use custom logger


# todo: parse browser interactions
class Heroku:
    # pandas dataframe with extracted data
    heroku_data = pd.DataFrame()
    save_p = False  # save data as pickle file
    load_p = False  # load data as pickle file
    save_csv = False  # save data as csv file
    # pandas dataframe with mapping
    mapping = pd.read_csv(tr.common.get_configs('mapping_stimuli'))
    # resolution for keypress data
    res = tr.common.get_configs('kp_resolution')
    # number of stimuli
    num_stimuli = tr.common.get_configs('num_stimuli')
    # number of stimuli shown for each participant
    num_stimuli_participant = tr.common.get_configs('num_stimuli_participant')
    # number of repeats for each stimulus
    num_repeat = tr.common.get_configs('num_repeat')
    # allowed number of stimuli with detected wrong duration
    allowed_length = tr.common.get_configs('allowed_stimuli_wrong_duration')
    # pickle file for saving data
    file_p = 'heroku_data.p'
    # csv file for saving data
    file_data_csv = 'heroku_data.csv'
    # csv file for saving data for images
    file_points_csv = 'points'
    # csv file for saving data for workers
    file_points_worker_csv = 'points_worker'
    # csv file for saving data for images for each duration
    file_points_duration_csv = 'points_duration'
    # csv file for mapping of stimuli
    file_mapping_csv = 'mapping.csv'
    # keys with meta information
    meta_keys = ['worker_code',
                 'browser_user_agent',
                 'browser_app_name',
                 'browser_major_version',
                 'browser_full_version',
                 'browser_name',
                 'window_height',
                 'window_width',
                 'video_ids',
                 'participant_group']
    # prefixes used for files in node.js implementation
    prefixes = {'stimulus': 'video_'}
    # stimulus duration
    default_dur = 0

    def __init__(self,
                 files_data: list,
                 save_p: bool,
                 load_p: bool,
                 save_csv: bool,
                 output_dir='output'):
        # list of files with raw data
        self.files_data = files_data
        # save data as pickle file
        self.save_p = save_p
        # load data as pickle file
        self.load_p = load_p
        # save data as csv file
        self.save_csv = save_csv
        # read in durations of stimuli from a config file
        self.hm_resolution_range = int(50000/tr.common.get_configs('hm_resolution'))
        self.num_stimuli = tr.common.get_configs('num_stimuli')
        self.output_dir = output_dir
        # Define subdirectories for batches and other outputs
        self.batch_dir = os.path.join(self.output_dir, 'batches')
        self.anova_dir = os.path.join(self.output_dir, 'anova_results')

        # Ensure directories exist
        os.makedirs(self.batch_dir, exist_ok=True)
        os.makedirs(self.anova_dir, exist_ok=True)
    def set_data(self, heroku_data):
        """Setter for the data object.
        """
        old_shape = self.heroku_data.shape  # store old shape for logging
        self.heroku_data = heroku_data
        logger.info('Updated heroku_data. Old shape: {}. New shape: {}.',
                    old_shape,
                    self.heroku_data.shape)

    def read_data(self, filter_data=True):
        """Read data into an attribute.

        Args:
            filter_data (bool, optional): flag for filtering data.

        Returns:
            dataframe: updated dataframe.
        """
        # load data
        if self.load_p:
            df = tr.common.load_from_p(self.file_p,
                                       'heroku data')
        # process data
        else:
            # read files with heroku data one by one
            data_list = []
            data_dict = {}  # dictionary with data
            for file in self.files_data:
                logger.info('Reading heroku data from {}.', file)
                f = open(file, 'r')
                # add data from the file to the dictionary
                data_list += f.readlines()
                f.close()
            # hold info on previous row for worker
            prev_row_info = pd.DataFrame(columns=['worker_code',
                                                  'time_elapsed'])
            prev_row_info.set_index('worker_code', inplace=True)
            # read rows in data
            for row in tqdm(data_list):  # tqdm adds progress bar
                # use dict to store data
                dict_row = {}
                # load data from a single row into a list
                list_row = json.loads(row)
                # last found stimulus
                stim_name = ''
                # trial last found stimulus
                stim_trial = -1
                # last time_elapsed for logging duration of trial and stimulus
                elapsed_l = 0
                elapsed_l_stim = 0
                # record worker_code in the row. assuming that each row has at
                # least one worker_code
                worker_code = [d['worker_code'] for d in list_row['data'] if 'worker_code' in d][0]
                if tr.common.get_configs('only_lab') == 1:
                    if re.search("lab_pp_", worker_code) is None:
                        continue
                # go over cells in the row with data
                for data_cell in list_row['data']:
                    # extract meta info form the call
                    for key in self.meta_keys:
                        if key in data_cell.keys():
                            # piece of meta data found, update dictionary
                            dict_row[key] = data_cell[key]
                            if key == 'worker_code':
                                logger.debug('{}: working with row with data.',
                                             data_cell['worker_code'])
                    # check if stimulus data is present
                    if 'stimulus' in data_cell.keys():
                        # record last timestamp before video
                        if 'black_frame.png' in data_cell['stimulus']:
                            # record timestamp at the black frame to compute
                            # the length of the stimulus
                            if 'time_elapsed' in data_cell.keys():
                                elapsed_l_stim = float(data_cell['time_elapsed'])
                        # extract name of stimulus after last slash
                        # list of stimuli. use 1st
                        if isinstance(data_cell['stimulus'], list):
                            stim_no_path = data_cell['stimulus'][0].rsplit('/', 1)[-1]
                        # single stimulus
                        else:
                            stim_no_path = data_cell['stimulus'].rsplit('/', 1)[-1]
                        # remove extension
                        stim_no_path = os.path.splitext(stim_no_path)[0]
                        # skip is videos from instructions
                        if 'video_test_' in stim_no_path:
                            continue
                        # Check if it is a block with stimulus and not an
                        # instructions block
                        if (tr.common.search_dict(self.prefixes, stim_no_path)
                                is not None):
                            # stimulus is found
                            logger.debug('Found stimulus {}.', stim_no_path)
                            if self.prefixes['stimulus'] in stim_no_path:
                                # Record that stimulus was detected for the
                                # cells to follow
                                stim_name = stim_no_path
                                # record trial of stimulus
                                stim_trial = data_cell['trial_index']
                                # add trial duration
                                if 'time_elapsed' in data_cell.keys():
                                    # positive time elapsed from last cell
                                    if elapsed_l_stim:
                                        time = elapsed_l_stim
                                    # non-positive time elapsed. use value from
                                    # the known cell for worker
                                    else:
                                        time = prev_row_info.loc[worker_code, 'time_elapsed']
                                    # calculate duration
                                    dur = float(data_cell['time_elapsed']) - time
                                    if stim_name + '-dur' not in dict_row.keys() and dur > 0:
                                        # first value
                                        dict_row[stim_name + '-dur'] = dur
                    # keypresses
                    if 'rts' in data_cell.keys() and stim_name != '':
                        # record given keypresses
                        responses = data_cell['rts']
                        logger.debug('Found {} points in keypress data.', len(responses))
                        # extract pressed keys and rt values
                        key = [point['key'] for point in responses]
                        rt = [point['rt'] for point in responses]
                        # check if values were recorded previously
                        if stim_name + '-key' not in dict_row.keys():
                            # first value
                            dict_row[stim_name + '-key'] = key
                        else:
                            # previous values found
                            dict_row[stim_name + '-key'].extend(key)
                        # check if values were recorded previously
                        if stim_name + '-rt' not in dict_row.keys():
                            # first value
                            dict_row[stim_name + '-rt'] = rt
                        else:
                            # previous values found
                            dict_row[stim_name + '-rt'].extend(rt)
                    # eye tracking data
                    if 'webgazer_data' in data_cell.keys() and stim_name != '':
                        # record eye tracking data
                        et_data = data_cell['webgazer_data']
                        logger.debug('Found {} points in eye tracking data.', len(et_data))
                        # extract x,y,t values
                        x = [point['x'] for point in et_data]
                        y = [point['y'] for point in et_data]
                        t = [point['t'] for point in et_data]
                        # check if values not already recorded
                        if stim_name + '-x' not in dict_row.keys():
                            # first value
                            dict_row[stim_name + '-x'] = x
                        else:
                            # previous values found
                            dict_row[stim_name + '-x'].extend(x)
                        # check if values not already recorded
                        if stim_name + '-y' not in dict_row.keys():
                            # first value
                            dict_row[stim_name + '-y'] = y
                        else:
                            # previous values found
                            dict_row[stim_name + '-y'].extend(y)
                        # check if values not already recorded
                        if stim_name + '-t' not in dict_row.keys():
                            # first value
                            dict_row[stim_name + '-t'] = t
                        else:
                            # previous values found
                            dict_row[stim_name + '-t'].extend(t)
                    # questions after stimulus
                    if ('response' in data_cell.keys() and stim_name != '' and
                       data_cell['response'] is not None):
                        # check if it is not dictionary
                        if 'slider-0' not in data_cell['response']:
                            continue
                        # record given answers
                        responses = data_cell['response']
                        logger.debug('Found responses to questions {}.', responses)
                        # unpack questions and answers
                        questions = []
                        answers = []
                        for key, value in responses.items():
                            questions.append(key)
                            answers.append(int(value))
                        # check if values were recorded previously
                        if stim_name + '-qs' not in dict_row.keys():
                            # first value
                            dict_row[stim_name + '-qs'] = questions
                        else:
                            # previous values found
                            dict_row[stim_name + '-qs'].extend(questions)
                        # Check if time spent values were recorded
                        # previously
                        if stim_name + '-as' not in dict_row.keys():
                            # first value
                            dict_row[stim_name + '-as'] = answers
                        else:
                            # previous values found
                            dict_row[stim_name + '-as'].extend(answers)
                    # browser interaction events
                    if 'interactions' in data_cell.keys() and stim_name != '':
                        interactions = data_cell['interactions']
                        logger.debug('Found {} browser interactions.', len(interactions))
                        # extract events and timestamps
                        event = []
                        time = []
                        for interation in interactions:
                            if interation['trial'] == stim_trial:
                                event.append(interation['event'])
                                time.append(interation['time'])
                        # Check if inputted values were recorded previously
                        if stim_name + '-event' not in dict_row.keys():
                            # first value
                            dict_row[stim_name + '-event'] = event
                        else:
                            # previous values found
                            dict_row[stim_name + '-event'].extend(event)
                        # check if values were recorded previously
                        if stim_name + '-time' not in dict_row.keys():
                            # first value
                            dict_row[stim_name + '-time'] = time
                        else:
                            # previous values found
                            dict_row[stim_name + '-time'].extend(time)
                    # sliders after experiment
                    if ('response' in data_cell.keys() and stim_name == '' and
                       data_cell['response'] is not None):
                        # check if it is not post-trial data
                        if 'slider-5' not in data_cell['response']:
                            continue
                        # record given keypresses
                        responses_end = data_cell['response']
                        logger.debug('Found responses to the questions in ' +
                                     'the end {}.', responses_end)
                        for key, value in responses_end.items():
                            # check if values not already recorded
                            if stim_name + 'end-' + key not in dict_row.keys():
                                # first value
                                dict_row['end-' + key] = value
                            else:
                                # previous values found
                                dict_row['end-' + key].extend(value)
                    # record last time_elapsed
                    if 'time_elapsed' in data_cell.keys():
                        elapsed_l = float(data_cell['time_elapsed'])
                # update last time_elapsed for worker
                prev_row_info.loc[dict_row['worker_code'], 'time_elapsed'] = elapsed_l
                # worker_code was encountered before
                if dict_row['worker_code'] in data_dict.keys():
                    # iterate over items in the data dictionary
                    for key, value in dict_row.items():
                        # worker_code does not need to be added
                        if key in self.meta_keys:
                            data_dict[dict_row['worker_code']][key] = value
                            continue
                        # new value
                        if key + '-0' not in data_dict[dict_row['worker_code']].keys():
                            data_dict[dict_row['worker_code']][key + '-0'] = value
                        # update old value
                        else:
                            # traverse repetition ids until get new repetition
                            for rep in range(0, self.num_repeat):
                                # build new key with id of repetition
                                new_key = key + '-' + str(rep)
                                if new_key not in data_dict[dict_row['worker_code']].keys():
                                    data_dict[dict_row['worker_code']][new_key] = value
                                    break
                # worker_code is encountered for the first time
                else:
                    # iterate over items in the data dictionary and add -0
                    for key, value in list(dict_row.items()):
                        # worker_code does not need to be added
                        if key in self.meta_keys:
                            continue
                        # new value
                        dict_row[key + '-0'] = dict_row.pop(key)
                    # add row of data
                    data_dict[dict_row['worker_code']] = dict_row
            # turn into pandas dataframe
            df = pd.DataFrame(data_dict)
            df = df.transpose()
            # report people that attempted study
            unique_worker_codes = df['worker_code'].drop_duplicates()
            logger.info('People who attempted to participate: {}', unique_worker_codes.shape[0])
            # filter data
            if filter_data:
                df = self.filter_data(df)
            # sort columns alphabetically
            df = df.reindex(sorted(df.columns), axis=1)
            # move worker_code to the front
            worker_code_col = df['worker_code']
            df.drop(labels=['worker_code'], axis=1, inplace=True)
            df.insert(0, 'worker_code', worker_code_col)
        # save to pickle
        if self.save_p:
            tr.common.save_to_p(self.file_p, df, 'heroku data')
        # save to csv
        if self.save_csv:
            # build path
            if not os.path.exists(tr.settings.output_dir):
                os.makedirs(tr.settings.output_dir)
            # save to file
            df.to_csv(os.path.join(tr.settings.output_dir, self.file_data_csv), index=False)
            logger.info('Saved heroku data to csv file {}', self.file_data_csv + '.csv')
        # update attribute
        self.heroku_data = df
        # return df with data
        return df

    def read_mapping(self):
        """
        Read mapping.
        """
        # read mapping from a csv file
        df = pd.read_csv(tr.common.get_configs('mapping_stimuli'))
        # set index as stimulus_id
        df.set_index('video_id', inplace=True)
        # update attribute
        self.mapping = df
        # return mapping as a dataframe
        return df

    def points(self, df, save_csv=True):
        """
        Create arrays with coordinates for images.
        save_points: save dictionary with points.
        if save_points:: save dictionary with points for each worker.
        """
        logger.info('Extracting coordinates for {} stimuli.', self.num_stimuli)
        # determining the set sample resolution for the heatmap animation
        hm_resolution = tr.common.get_configs('hm_resolution')
        # dictionaries to store points
        points = {}
        points_worker = {}
        points_duration = [{} for x in range(0, 5000000000, hm_resolution)]
        # window values for normalization
        height = int(tr.common.get_configs('stimulus_height'))
        width = int(tr.common.get_configs('stimulus_width'))
        # allowed percentage of codeblocks in the middle
        allowed_percentage = 0.2
        area = 100
        # calculate the middle of the stimulus
        width_middle = round(width/2)
        height_middle = round(height/2)
        # polygon for the centre
        polygon = Polygon([(width_middle - area, height_middle + area),
                           (width_middle + area, height_middle + area),
                           (width_middle - area, height_middle - area),
                           (width_middle + area, height_middle - area)])
        # loop over stimuli from 1 to self.num_stimuli
        # tqdm adds progress bar
        for id_video in tqdm(range(0, self.num_stimuli)):
            # create empty list to store points for the stimulus
            points[id_video] = []
            # loop over durations of stimulus
            dur = self.mapping.loc[id_video]['video_length']
            number_dur = len(range(0, dur, hm_resolution))
            for duration in range(0, number_dur):
                # create empty list to store points for the stimulus of given
                # duration
                points_duration[duration][id_video] = []
                # create empty list to store points of given duration for the
                # stimulus
                # build names of columns in df
                x = 'video_'+str(id_video)+'-x-0'
                y = 'video_'+str(id_video)+'-y-0'
                t = 'video_'+str(id_video)+'-t-0'

                if x not in df.keys() or y not in df.keys():
                    logger.debug('Indices not found: {} or {}.', x, y)
                    continue
                # trim df
                stim_from_df = df[[x, y, t]]
                # iterate of data from participants for the given stimulus
                for pp in range(len(stim_from_df)):
                    # input given by participant
                    given_y = stim_from_df.iloc[pp][y]
                    given_x = stim_from_df.iloc[pp][x]
                    given_t = stim_from_df.iloc[pp][t]
                    # normalize window size among pp
                    pp_height = int(df.iloc[pp]['window_height'])
                    pp_width = int(df.iloc[pp]['window_width'])
                    norm_y = height/pp_height
                    norm_x = width/pp_width
                    # detected percentage of codeblocks in the middle
                    detected = 0
                    # skip if no points for worker
                    if type(given_y) is list:
                        # Check if imput from stimulus isn't blank
                        if given_x != []:
                            length_points = len(given_y)
                            for val in range(length_points-1):
                                # convert to point object
                                point = Point(given_x[val]*norm_x,
                                              given_y[val]*norm_y)

                                # check if point is within a polygon in the middle
                                if polygon.contains(point):
                                    # point in the middle detected
                                    detected += 1
                                # Check if for the worker there were more than
                                # allowed limit of points in the middle
                                if detected / length_points > allowed_percentage:
                                    break
                            if detected / length_points < allowed_percentage:
                                for value in range(length_points):
                                    t_step = round(given_t[value]/hm_resolution)
                                    if duration == t_step:
                                        if id_video not in points_duration[duration]:
                                            points_duration[duration][id_video] = [[given_x[value]*norm_x,
                                                                                    given_y[value]*norm_y]]
                                        else:
                                            points_duration[duration][id_video].append([given_x[value]*norm_x,
                                                                                        given_y[value]*norm_y])
                                    if duration < t_step:
                                        break
                                    # start adding points to the points_duration list
                                # iterate over all values given by the participand
                                # for val in range(len(given_y)-1):
                                #     # add coordinates
                                #     if id_video not in points:
                                #         points[id_video] = [[(coords[0]),
                                #                             (coords[1])]]
                                #     else:
                                #         points[id_video].append([(coords[0]),
                                #                                 (coords[1])])
                                    # if stim_from_df.index[pp] not in points_worker:
                                    #     points_worker[stim_from_df.index[pp]] = [[(coords[0]),
                                    #                                              (coords[1])]]
                                    # else:
                                    #     points_worker[stim_from_df.index[pp]].append([(coords[0]),
                                    #                                                   (coords[1])])
        # save to csv
        if save_csv:
            # # all points for each image
            # # create a dataframe to save to csv
            # df_csv = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in points.items()]))
            # df_csv = df_csv.transpose()
            # # save to csv
            # df_csv.to_csv(tr.settings.output_dir + '/' +
            #               self.file_points_csv + '.csv')
            # logger.info('Saved dictionary of points to csv file {}.csv',
            #             self.file_points_csv)
            # all points for each worker
            # create a dataframe to save to csv
            # df_csv = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in points_worker.items()]))
            # df_csv = df_csv.transpose()
            # # save to csv
            # df_csv.to_csv(tr.settings.output_dir + '/' +
            #               self.file_points_worker_csv + '.csv')
            # logger.info('Saved dictionary of points for each worker to csv ' +
            #             'file {}.csv',
            #             self.file_points_worker_csv)
            # points for each image for each stimulus duration
            # create a dataframe to save to csv
            for duration in range(0, self.hm_resolution_range):
                try:
                    df_csv = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in points_duration[duration].items()]))
                except KeyError:
                    break
                df_csv = df_csv.transpose()
                # save to csv
                df_csv.to_csv(tr.settings.output_dir +
                              '/' +
                              self.file_points_duration_csv +
                              '_' +
                              str(duration) +
                              '.csv')
                logger.info('Saved dictionary of points for duration {} ' +
                            'to csv file {}_{}.csv',
                            str(duration),
                            self.file_points_duration_csv,
                            str(duration))
        # return points
        return points, points_worker, points_duration

    def process_kp(self, filter_length=True):
        """Process keypresses for resolution self.res.
        Returns:
            mapping: updated mapping df.
        Args:
            filter_length (bool, optional): filter out stimuli with unexpected length.
        """
        logger.info('Processing keypress data with res={} ms.', self.res)
        # array to store all binned rt data
        mapping_rt = []
        # array to store all raw binned rt data per pp
        mapping_rt_raw = []
        # counter of videos filtered because of length
        counter_filtered = 0
        # loop through all stimuli
        # todo: account for multiple repetitions
        for num in tqdm(range(self.num_stimuli)):
            # video ID
            video_id = 'V' + str(num)
            # extract video length
            video_len = self.mapping.loc[video_id]['video_length']
            # add new row to df with raw data
            video_kp = []
            video_kp_raw = []
            # df to store keypresses in bins per pp for this individual stimulus with array of zeros to be able to
            # store counters for multiple repetitions
            cols = len(self.heroku_data.index)
            rows = len(list(range(self.res, video_len + self.res, self.res)))
            pp_kp = pd.DataFrame([[np.zeros(self.num_repeat) * self.num_repeat for _ in range(cols)] for _ in range(rows)],  # noqa: E501
                                 index=list(range(self.res, video_len + self.res, self.res)),
                                 columns=self.heroku_data.index)
            # go over repetitions
            for rep in range(self.num_repeat):
                # add suffix with repetition ID
                video_rt = 'video_' + str(num) + '-rt-' + str(rep)
                video_dur = 'video_' + str(num) + '-dur-' + str(rep)
                rt_data = []
                counter_data = 0
                for (col_name, col_data) in self.heroku_data.items():
                    # find the right column to loop through
                    if video_rt == col_name:
                        # loop through rows in column
                        for row_index, row in enumerate(col_data):
                            # consider only videos of allowed length
                            if video_dur in self.heroku_data.keys() and filter_length:
                                # extract recorded duration
                                dur = self.heroku_data.iloc[row_index][video_dur]
                                # check if duration is within limits
                                if dur < self.mapping['min_dur'][video_id] or dur > self.mapping['max_dur'][video_id]:
                                    # increase counter of filtered videos
                                    logger.debug('Filtered keypress data from video {} of detected duration of {} for '
                                                 + 'worker {}.',
                                                 video_id, dur,
                                                 self.heroku_data.index[row_index])
                                    # increase counter of filtered videos
                                    counter_filtered = counter_filtered + 1
                                    continue
                            # check if data is string to filter out nan data
                            if type(row) is list:
                                # saving amount of times the video has been watched
                                counter_data = counter_data + 1
                                # if list contains only one value, append to rt_data
                                if len(row) == 1:
                                    rt_data.append(row[0])
                                    # record raw value for pp
                                    for rt_bin in range(self.res, video_len + self.res, self.res):
                                        if rt_bin - self.res < row[0] <= rt_bin:
                                            pp_kp.at[rt_bin, self.heroku_data.index[row_index]][rep] = 1
                                # if list contains more then one value, go  through list to remove keyholds
                                elif len(row) > 1:
                                    # fill data gap for the first half a second (0.5 s) of holding the key
                                    if row[0] <= 540:  # If the first button press is 'exactly' at 0.5 seconds
                                        row = np.concatenate((np.arange(40, row[0] + 40, 40), row[1:]))
                                    # find indexes with a gap of 'exactly' 0.5 seconds
                                    gap_indexes = 1 + np.where((np.diff(row) >= 420) & (np.diff(row) <= 540))[0]
                                    # loop over all gaps in backward order
                                    for k in reversed(range(len(gap_indexes))):
                                        index = gap_indexes[k]
                                        filled_gap = np.arange(row[index - 1] + 40, row[index] + 40, 40)
                                        row = np.concatenate((row[:index], filled_gap, row[index:]))
                                    # # normalise the keypress data
                                    # row = self.mapping['video_length'][video_id] * row / (trial_duration / 1000)
                                    # go over prepared keypress data
                                    for j in range(1, len(row)):
                                        # todo: old version. remove when KP code is stable
                                        # # if time between 2 stimuli is more than 35 ms, add to array (no hold)
                                        # if row[j] - row[j - 1] > 35:
                                        # append button press data to rt array
                                        rt_data.append(row[j])
                                        # record raw value for pp
                                        for rt_bin in range(self.res, video_len + self.res, self.res):
                                            if rt_bin - self.res < row[j] <= rt_bin:
                                                pp_kp.at[rt_bin, self.heroku_data.index[row_index]][rep] = 1
                                # if list contains more then one value, go  through list to remove keyholds
                        # print(self.heroku_data.index[row_index], video_id, pp_kp.sum())
                        # print(len(pp_kp.index))
                        # # calculate counts
                        # data = (pp_kp.sum() / len(pp_kp.index) * 100).tolist()
                        # print(data)
                        # # remove tail after video duration
                        # data = data[0:int(self.mapping['video_length'][video_id] / self.res)]
                        # video_kp.append(data)
                        # todo: old version. remove when KP code is stable
                        # # if all data for one video was found, divide them in bins
                        # kp = []
                        # # loop over all bins, dependent on resolution
                        # bin_counter = 0  # record number of rt values found within bin
                        # for rt in range(self.res, video_len + self.res, self.res):
                        #     bin_counter = 0
                        #     for data in rt_data:
                        #         # go through all video data to find all data within specific bin
                        #         if rt - self.res < data <= rt:
                        #             # if data is found, up bin counter
                        #             bin_counter = bin_counter + 1
                        #     if counter_data:
                        #         percentage = bin_counter / counter_data
                        #         kp.append(round(percentage * 100))
                        #         print(bin_counter, counter_data, percentage)
                        #     else:
                        #         kp.append(0)
                        # # store keypresses from repetition
                        # video_kp.append(kp)
                        # calculate sums for counters from multiple iterations
                        pp_kp = pp_kp.map(lambda x: int(sum(x)) if isinstance(x, np.ndarray) else x)
                        # calculate counts
                        data = (pp_kp.sum(axis=1) / len(pp_kp.index) / self.num_repeat * 100).tolist()
                        # remove tail after video duration
                        data = data[0:int(self.mapping['video_length'][video_id] / self.res)]
                        video_kp.append(data)
                        # store raw data from repetition
                        # todo: fix extra [] added to results
                        video_kp_raw.append(pp_kp.values.tolist())
                        break
            # calculate mean keypresses from all repetitions
            kp_mean = [*map(mean, zip(*video_kp))]
            # append data from one video to the mapping array
            mapping_rt.append(kp_mean)
            # todo: raw data does not take multiple repetitions into account
            mapping_rt_raw.append(video_kp_raw)
        if filter_length:
            logger.info('Filtered out keypress data from {} videos with unexpected length.', counter_filtered)
        # update own mapping to include keypress data
        self.mapping['kp'] = mapping_rt
        self.mapping['kp_raw'] = mapping_rt_raw
        # save to csv
        if self.save_csv:
            # save to csv
            self.mapping.to_csv(os.path.join(tr.settings.output_dir, self.file_mapping_csv))
        # return new mapping
        return self.mapping

    def process_kp_to_batches(self, output_dir=None, filter_length=True):

        if output_dir is None:
            output_dir = tr.settings.output_dir  # Default output directory
        logger.info('Processing keypress data into 21 video batches with res={} ms.'.format(self.res))
        os.makedirs(output_dir, exist_ok=True)

        self.heroku_data['EgoCar'] = self.heroku_data['participant_group'].map(lambda x: 0 if x in [0, 1] else 1)
        video_batches = [[i, i + 21, i + 42, i + 63] for i in range(21)]

        for batch_num, videos in enumerate(video_batches):
            logger.info(f'Processing batch {batch_num} for videos: {videos}')
            batch_data = []

            for video_num in videos:
                video_id = f'V{video_num}'
                if video_id not in self.mapping.index:
                    logger.warning(f"Video {video_id} not found in mapping. Skipping...")
                    continue

                video_len = self.mapping.loc[video_id]['video_length']

                for rep in range(self.num_repeat):
                    video_rt = f'video_{video_num}-rt-{rep}'
                    video_dur = f'video_{video_num}-dur-{rep}'

                    for row_index, row in self.heroku_data.iterrows():
                        participant_id = row.name
                        ego_car = row['EgoCar']
                        target_car = 0 if (video_num % 2 == 0) else 1

                        if ((ego_car == 0 and video_num >= 42) or
                            (ego_car == 1 and video_num < 42)):
                            continue

                        if video_rt not in self.heroku_data.columns:
                            continue

                        rt_data = row[video_rt]
                        if not isinstance(rt_data, list):
                            continue

                        if video_dur in self.heroku_data.columns and filter_length:
                            dur = row[video_dur]
                            if dur < self.mapping['min_dur'][video_id] or dur > self.mapping['max_dur'][video_id]:
                                logger.debug(f"Filtered video {video_id} duration {dur} for worker {participant_id}.")
                                continue

                        for rt_bin in range(self.res, video_len + self.res, self.res):
                            kp_count = sum(1 for rt in rt_data if rt_bin - self.res < rt <= rt_bin)
                            batch_data.append({
                                'ParticipantID': participant_id,
                                'EgoCar': ego_car,
                                'TargetCar': target_car,
                                'VideoNumber': video_num,
                                'TimeBin': rt_bin,
                                'KPNumber': kp_count
                            })

            # Create a DataFrame
            batch_df = pd.DataFrame(batch_data)
            # Map TimeBin to TimeIndex based on sorted order
            if 'TimeBin' in batch_df.columns:
                time_bin_mapping = {value: index for index, value in enumerate(sorted(batch_df['TimeBin'].unique()))}
                batch_df['TimeIndex'] = batch_df['TimeBin'].map(time_bin_mapping)
            else:
                logger.warning(f"'TimeBin' column is missing in batch {batch_num}. Skipping TimeIndex generation.")


            # Ensure VideoNumber is in the DataFrame
            if 'VideoNumber' not in batch_df.columns:
                logger.error(f"'VideoNumber' column is missing in batch {batch_num}. Skipping batch.")
                continue

            # Remove incomplete data
            batch_df = batch_df.dropna(subset=['TimeBin', 'KPNumber'])

            # Add TimeIndex
            logger.info(f"Adding TimeIndex to batch {batch_num}.")
            unique_bins = sorted(batch_df['TimeBin'].unique())  # Sort TimeBin values
            bin_to_index = {time_bin: idx for idx, time_bin in enumerate(unique_bins)}  # Map TimeBin to sequential indices
            batch_df['TimeIndex'] = batch_df['TimeBin'].map(bin_to_index)  # Add TimeIndex column

            # Validate and save batch data
            valid_data = []
            for time_bin in batch_df['TimeBin'].unique():
                time_bin_data = batch_df[batch_df['TimeBin'] == time_bin]
                group_counts = time_bin_data.groupby(['EgoCar', 'TargetCar']).size()
                if group_counts.min() > 1:  # Ensure at least 2 participants per group-condition combination
                    valid_data.append(time_bin_data)

            if valid_data:
                final_df = pd.concat(valid_data)
                final_df['EgoCar'] = final_df['EgoCar'].astype('category')
                final_df['TargetCar'] = final_df['TargetCar'].astype('category')

                # Save the batch data if self.save_csv is True

                if self.save_csv:
                    batch_file = os.path.join(tr.settings.output_dir, f'batch_{batch_num}_keypress_data.csv')
                    final_df.to_csv(batch_file, index=False)
                    logger.info(f"Batch {batch_num} data saved to {batch_file}.")
            else:
                logger.warning(f"No valid data for batch {batch_num}. File not created.")



    def calculate_descriptive_statistics(self, batch_dir, output_dir):
        """
        Calculate descriptive statistics, Kolmogorov-Smirnov, and Levene's test for each batch file.

        Args:
            batch_dir (str): Directory containing batch files.
            output_dir (str): Directory to save descriptive statistics files.

        Returns:
            None
        """
        os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists

        for batch_file in sorted(os.listdir(batch_dir)):
            if not batch_file.startswith('batch_') or not batch_file.endswith('.csv'):
                continue

            batch_path = os.path.join(batch_dir, batch_file)

            # Check if the file is empty
            if os.path.getsize(batch_path) == 0:
                logger.warning(f"Skipping empty batch file: {batch_file}")
                continue

            try:
                batch_data = pd.read_csv(batch_path)
                logger.debug(f"Processing {batch_file}: Columns: {batch_data.columns}")
            except Exception as e:
                logger.error(f"Failed to read batch file {batch_file}: {e}")
                continue

            if 'VideoNumber' not in batch_data.columns or 'TimeBin' not in batch_data.columns:
                logger.error(f"Missing required columns in {batch_file}. Available columns: {batch_data.columns}")
                continue

            logger.info(f"Processing {batch_file} for descriptive statistics...")

            descriptive_stats = []
            for (video_num, time_bin), group_data in batch_data.groupby(['VideoNumber', 'TimeBin']):
                if group_data['KPNumber'].isnull().all():
                    logger.warning(f"No valid data for Video {video_num}, TimeBin {time_bin}. Skipping...")
                    continue

                stats = {
                    'VideoNumber': video_num,
                    'TimeBin': time_bin,
                    'mean': group_data['KPNumber'].mean(),
                    'std': group_data['KPNumber'].std(),
                    'min': group_data['KPNumber'].min(),
                    'max': group_data['KPNumber'].max(),
                    'count': group_data['KPNumber'].count()
                }

                # Debugging input data for Kolmogorov-Smirnov test
                logger.debug(f"Data for Kolmogorov-Smirnov test: {group_data['KPNumber']}")

                try:
                    if len(group_data) > 3 and stats['std'] > 0:  # At least 4 data points and non-zero std
                        # Test against a normal distribution with the same mean and std
                        theoretical_cdf = norm(loc=stats['mean'], scale=stats['std']).cdf
                        ks_stat, p_value = ks_1samp(group_data['KPNumber'], theoretical_cdf)
                        # Ensure p-value is in range [0, 1]
                        if not (0 <= p_value <= 1):
                            logger.warning(f"Invalid KS p-value for Video {video_num}, TimeBin {time_bin}: {p_value}")
                            p_value = None  # Reset invalid p-values
                        stats['KS-stat'] = ks_stat
                        stats['KS-p'] = p_value
                    else:
                        stats['KS-stat'], stats['KS-p'] = None, None
                except Exception as e:
                    logger.warning(f"Kolmogorov-Smirnov test failed for Video {video_num}, TimeBin {time_bin}: {e}")
                    stats['KS-stat'], stats['KS-p'] = None, None

                # Debugging input data for Levene's test
                logger.debug(f"Data for Levene's Test: {group_data}")

                try:
                    group_values = [
                        group['KPNumber'].values
                        for _, group in group_data.groupby('EgoCar')
                        if len(group) > 1
                    ]
                    if len(group_values) > 1:
                        levene_stat, levene_p = levene(*group_values)
                        stats['Levene-stat'] = levene_stat
                        stats['Levene-p'] = levene_p
                    else:
                        stats['Levene-stat'], stats['Levene-p'] = None, None
                except Exception as e:
                    logger.warning(f"Levene's test failed for Video {video_num}, TimeBin {time_bin}: {e}")
                    stats['Levene-stat'], stats['Levene-p'] = None, None

                descriptive_stats.append(stats)

            if descriptive_stats:
                descriptive_file = os.path.join(output_dir, f"{batch_file.replace('.csv', '_descriptive_statistics.csv')}")
                pd.DataFrame(descriptive_stats).to_csv(descriptive_file, index=False)
                logger.info(f"Descriptive statistics saved to {descriptive_file}.")
            else:
                logger.warning(f"No descriptive statistics generated for {batch_file}.")


    def process_stimulus_questions(self, questions):
        """Process questions that follow each stimulus.

        Args:
            questions (list): list of questions with types of possible values
                              as int or str.

        Returns:
            dataframe: updated mapping dataframe.
        """
        logger.info('Processing post-stimulus questions')
        # array in which arrays of video_as data is stored
        mapping_as = []
        # loop through all stimuli
        for num in tqdm(range(self.num_stimuli)):
            # calculate length of array with answers
            length = 0
            for q in questions:
                # 1 column required for numeric data
                # numeric answer, create 1 column to store mean value
                if q['type'] == 'num':
                    length = length + 1
                # strings as answers, create columns to store counts
                elif q['type'] == 'str':
                    length = length + len(q['options'])
                else:
                    logger.error('Wrong type of data {} in question {} provided.', q['type'], q['question'])
                    return -1
            # array in which data of a single stimulus is stored
            answers = [[[] for i in range(self.heroku_data.shape[0])] for i in range(len(questions))]
            # for number of repetitions in survey, add extra number
            for rep in range(self.num_repeat):
                # add suffix with repetition ID
                video_as = 'video_' + str(num) + '-as-' + str(rep)
                video_order = 'video_' + str(num) + '-qs-' + str(rep)
                # loop over columns
                for col_name, col_data in self.heroku_data.items():
                    # when col_name equals video, then check
                    if col_name == video_as:
                        # loop over rows in column
                        for pp, row in enumerate(col_data):
                            # filter out empty values
                            if type(row) is list:
                                order = self.heroku_data.iloc[pp][video_order]
                                # check if injection question is present
                                if 'injection' in order:
                                    # delete injection
                                    del row[order.index('injection')]
                                    del order[order.index('injection')]
                                # loop through questions
                                for i, q in enumerate(questions):
                                    # extract answer
                                    ans = row[order.index(q['question'])]
                                    # store answer from repetition
                                    answers[i][pp].append(ans)
            # calculate mean answers from all repetitions for numeric questions
            for i, q in enumerate(questions):
                if q['type'] == 'num' and answers[i]:
                    # convert to float
                    answers[i] = [list(map(float, sublist))
                                  for sublist in answers[i]]
                    # calculate mean of mean of responses of each participant
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore', category=RuntimeWarning)
                        answers[i] = np.nanmean([np.nanmean(j) for j in answers[i]])
            # save question data in array
            mapping_as.append(answers)
        # add column with data to current mapping file
        for i, q in enumerate(questions):
            # extract answers for the given question
            q_ans = [item[i] for item in mapping_as]
            # for numeric question, add column with mean values
            if q['type'] == 'num':
                self.mapping[q['question']] = q_ans
            # for textual question, add columns with counts of each value
            else:
                # go over options and count answers with the option for each
                # stimulus
                for option in q['options']:
                    # store counts in list
                    count_option = []
                    # go over each answer
                    for ans in q_ans:
                        # flatten list of answers
                        ans = [item for sublist in ans for item in sublist]
                        # add count for answers for the given option
                        count_option.append(ans.count(option))
                    # build name of column
                    col_name = q['question'] + '-' + option.replace(' ', '_')
                    col_name = col_name.lower()
                    # add to mapping
                    self.mapping[col_name] = count_option
        # save to csv
        if self.save_csv:
            # save to csv
            self.mapping.to_csv(os.path.join(tr.settings.output_dir, self.file_mapping_csv))
        # return new mapping
        return self.mapping

    def process_questions_to_batches(self, questions, output_dir=None, filter_length=True):
        """
        Process post-stimulus questions into video batches.

        Args:
            questions (list): List of question definitions (e.g., {'question': 'slider-0', 'type': 'num'}).
            filter_length (bool): Whether to filter videos by their duration.

        Returns:
            None
        """
        if output_dir is None:
            output_dir = tr.settings.output_dir  # Default output directory
        logger.info('Processing question data into video batches.')
        os.makedirs(output_dir, exist_ok=True)

        self.heroku_data['EgoCar'] = self.heroku_data['participant_group'].map(lambda x: 0 if x in [0, 1] else 1)
        video_batches = [[i, i + 21, i + 42, i + 63] for i in range(21)]

        for batch_num, videos in enumerate(video_batches):
            logger.info(f'Processing batch {batch_num} for videos: {videos}')
            batch_data = []

            for video_num in videos:
                video_id = f'V{video_num}'
                if video_id not in self.mapping.index:
                    logger.warning(f"Video {video_id} not found in mapping. Skipping...")
                    continue

                for rep in range(self.num_repeat):
                    video_as = f'video_{video_num}-as-{rep}'
                    video_order = f'video_{video_num}-qs-{rep}'

                    for row_index, row in self.heroku_data.iterrows():
                        participant_id = row.name
                        ego_car = row['EgoCar']
                        target_car = 0 if (video_num % 2 == 0) else 1

                        if ((ego_car == 0 and video_num >= 42) or
                            (ego_car == 1 and video_num < 42)):
                            continue

                        if video_as not in self.heroku_data.columns or video_order not in self.heroku_data.columns:
                            continue

                        answers = row[video_as]
                        order = row[video_order]

                        if not isinstance(answers, list) or not isinstance(order, list):
                            continue

                        if 'injection' in order:
                            del answers[order.index('injection')]
                            del order[order.index('injection')]

                        question_data = {}
                        for q in questions:
                            question_name = q['question']  # Extract the question name
                            if question_name in order:
                                idx = order.index(question_name)
                                question_data[question_name] = answers[idx]
                            else:
                                question_data[question_name] = None

                        batch_data.append({
                            'ParticipantID': participant_id,
                            'EgoCar': ego_car,
                            'TargetCar': target_car,
                            'VideoNumber': video_num,
                            **question_data
                        })

            # Create a DataFrame
            batch_df = pd.DataFrame(batch_data)

            # Ensure VideoNumber is in the DataFrame
            if 'VideoNumber' not in batch_df.columns:
                logger.error(f"'VideoNumber' column is missing in batch {batch_num}. Skipping batch.")
                continue

            # Remove rows with all NaN question data
            question_columns = [q['question'] for q in questions if q['question'] in batch_df.columns]
            batch_df = batch_df.dropna(subset=question_columns, how='all')

            # Validate and save batch data
            valid_data = []
            for video_num in batch_df['VideoNumber'].unique():
                video_data = batch_df[batch_df['VideoNumber'] == video_num]
                group_counts = video_data.groupby(['EgoCar', 'TargetCar']).size()
                if group_counts.min() > 1:  # Ensure at least 2 participants per group-condition combination
                    valid_data.append(video_data)

            if valid_data:
                final_df = pd.concat(valid_data)
                final_df['EgoCar'] = final_df['EgoCar'].astype('category')
                final_df['TargetCar'] = final_df['TargetCar'].astype('category')
                if self.save_csv:
                        batch_file = os.path.join(tr.settings.output_dir, f'batch_{batch_num}_poststimulus_data.csv')
                        final_df.to_csv(batch_file, index=False)
                        logger.info(f"Batch {batch_num} question data saved to {batch_file}.")
            else:
                logger.warning(f"No valid data for batch {batch_num}. File not created.")



    def filter_data(self, df):
        """
        Filter data.
        Args:
            df (dataframe): dataframe with data.


        Returns:
            dataframe: updated dataframe.
            centre bais
        """
        logger.info('Filtering heroku data.')
        # 1. People who made mistakes in injected questions
        # TODO: check for large lengths of videos.
        logger.info('Filter-h1. People who had too many stimuli of unexpected length.')
        # df to store data to filter out
        df_1 = pd.DataFrame()
        # array to store in video names
        video_dur = []
        for i in range(0, self.num_stimuli):
            for rep in range(0, self.num_repeat):
                video_dur.append('video_' + str(i) + '-dur-' + str(rep))
        # tqdm adds progress bar
        # loop over participants in data
        for index, row in tqdm(df.iterrows(), total=df.shape[0]):
            data_count = 0
            counter_filtered = 0
            for i in range(self.num_stimuli):
                for rep in range(self.num_repeat):
                    # add suffix with repetition ID
                    video_dur = 'video_' + str(i) + '-dur-' + str(rep)
                    # check id value is present
                    if video_dur not in row.keys():
                        continue
                    # check for nan values
                    if pd.isna(row[video_dur]):
                        continue
                    else:
                        # up data count when data is found
                        data_count = data_count + 1
                        if (row[video_dur] < (self.mapping['min_dur'].iloc[i])
                           or row[video_dur] > (self.mapping['max_dur'].iloc[i])):
                            # up counter if data with wrong length is found
                            counter_filtered = counter_filtered + 1
            # Only check for participants that watched all videos
            if data_count >= self.num_stimuli_participant * self.num_repeat:
                # check threshold ratio
                if counter_filtered / data_count > self.allowed_length:
                    # if threshold reached, append data of this participant to
                    # df_1
                    df_1 = pd.concat([df_1, pd.DataFrame([row])],
                                     ignore_index=True)
        logger.info('Filter-h1. People who had more than {} share of stimuli of unexpected length: {}.',
                    self.allowed_length,
                    df_1.shape[0])
        old_size = df.shape[0]
        df_filtered = pd.concat([df_1])
        # check if there are people to filter
        if not df_filtered.empty:
            # drop rows with filtered data
            unique_worker_codes = df_filtered['worker_code'].drop_duplicates()
            df = df[~df['worker_code'].isin(unique_worker_codes)]
            # reset index in dataframe
            df = df.reset_index()
        logger.info('Filtered in total in heroku data: {}.',
                    old_size - df.shape[0])
        return df

    def show_info(self):
        """
        Output info for data in object.
        """
        logger.info('No info to show.')
