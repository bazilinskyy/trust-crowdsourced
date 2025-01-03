# by Pavlo Bazilinskyy <pavlo.bazilinskyy@gmail.com>
import matplotlib.pyplot as plt
import matplotlib._pylab_helpers
from tqdm import tqdm
import os
import trust as tr
# from statistics import mean
# import pandas as pd
import re
from statistics import mean
tr.logs(show_level='info', show_color=True)
logger = tr.CustomLogger(__name__)  # use custom logger

# const
# SAVE_P = True  # save pickle files with data
# LOAD_P = False  # load pickle files with data
# SAVE_CSV = True  # load csv files with data
# FILTER_DATA = True  # filter Appen and heroku data
# CLEAN_DATA = True  # clean Appen data
# REJECT_CHEATERS = False  # reject cheaters on Appen
# CALC_COORDS = False  # extract points from heroku data
# UPDATE_MAPPING = True  # update mapping with keypress data
# SHOW_OUTPUT = True  # should figures be plotted
# SHOW_OUTPUT_KP = False  # should figures with keypress data be plotted
# SHOW_OUTPUT_ST = False  # should figures with stimulus data be plotted
# SHOW_OUTPUT_PP = False  # should figures with info about participants be plotted
# SHOW_OUTPUT_ET = False  # should figures for eye tracking be plotted

# for debugging, skip processing
SAVE_P = False  # save pickle files with data
LOAD_P = True  # load pickle files with data
SAVE_CSV = True  # load csv files with data
FILTER_DATA = False  # filter Appen and heroku data
CLEAN_DATA = False  # clean Appen data
REJECT_CHEATERS = False  # reject cheaters on Appen
CALC_COORDS = False  # extract points from heroku data
UPDATE_MAPPING = True  # update mapping with keypress data
SHOW_OUTPUT = True  # should figures be plotted
SHOW_OUTPUT_KP = True  # should figures with keypress data be plotted
SHOW_OUTPUT_ST = False  # should figures with stimulus data be plotted
SHOW_OUTPUT_PP = False  # should figures with info about participants be plotted
SHOW_OUTPUT_ET = False  # should figures for eye tracking be plotted

# todo: code for eye gaze analysis does not run on mac

file_mapping = 'mapping.p'  # file to save updated mapping
file_coords = 'coords.p'  # file to save lists with coordinates

if __name__ == '__main__':
    # create object for working with heroku data
    files_heroku = tr.common.get_configs('files_heroku')
    heroku = tr.analysis.Heroku(files_data=files_heroku,
                                save_p=SAVE_P,
                                load_p=LOAD_P,
                                save_csv=SAVE_CSV)
    # read heroku data
    heroku_data = heroku.read_data(filter_data=FILTER_DATA)
    # directly count participants in each group
    if 'participant_group' in heroku_data.columns:
        group_counts = heroku_data['participant_group'].value_counts()
        logger.info('Participant counts by group:')
        for group, count in group_counts.items():
            logger.info("Group {}: {} participants", group, count)
    else:
        logger.error("'participant_group' column not found in the data.")
    # create object for working with appen data
    file_appen = tr.common.get_configs('file_appen')
    appen = tr.analysis.Appen(file_data=file_appen,
                              save_p=SAVE_P,
                              load_p=LOAD_P,
                              save_csv=SAVE_CSV)
    # read appen data
    appen_data = appen.read_data(filter_data=FILTER_DATA,
                                 clean_data=CLEAN_DATA)
    # read frames
    # get keys in data files
    heroku_data_keys = heroku_data.keys()
    appen_data_keys = appen_data.keys()
    # flag and reject cheaters
    if REJECT_CHEATERS:
        qa = tr.analysis.QA(
            file_cheaters=tr.common.get_configs('file_cheaters'),
            job_id=tr.common.get_configs('appen_job'))
        qa.reject_users()
        qa.ban_users()
    # merge heroku and appen dataframes into one
    all_data = heroku_data.merge(appen_data,
                                 left_on='worker_code',
                                 right_on='worker_code')
    logger.info('Data from {} participants included in analysis.',
                all_data.shape[0])
    # update original data files
    if tr.common.get_configs('only_lab') == 0:
        heroku_data = all_data[all_data.columns.intersection(heroku_data_keys)]
        appen_data = all_data[all_data.columns.intersection(appen_data_keys)]
    heroku_data = heroku_data.set_index('worker_code')
    heroku.set_data(heroku_data)  # update object with filtered data
    appen_data = appen_data.set_index('worker_code')
    appen.set_data(appen_data)  # update object with filtered data
    appen.show_info()  # show info for filtered data
    # generate country-specific data
    countries_data = appen.process_countries()
    # create arrays with coordinates for stimuli
    if CALC_COORDS:
        points, _, points_duration = heroku.points(heroku_data)
        tr.common.save_to_p(file_coords,
                            [points, points_duration],
                            'points data')
    else:
        points, points_duration = tr.common.load_from_p(file_coords,
                                                        'points data')
    # update mapping with keypress data
    if UPDATE_MAPPING:
        # read in mapping of stimuli
        mapping = heroku.read_mapping()
        # process keypresses and update mapping
        mapping = heroku.process_kp(filter_length=False)
        # post-trial questions to process
        questions = [{'question': 'slider-0', 'type': 'num'},
                     {'question': 'slider-1', 'type': 'num'},
                     {'question': 'slider-2', 'type': 'num'}]
        # process post-trial questions and update mapping
        mapping = heroku.process_stimulus_questions(questions)
        # rename columns with responses to post-stimulus questions to meaningful names
        mapping = mapping.rename(columns={'slider-0': 'comfort',
                                          'slider-1': 'safety',
                                          'slider-2': 'expectation'})
        # export to pickle
        tr.common.save_to_p(file_mapping,  mapping, 'mapping of stimuli')
    else:
        mapping = tr.common.load_from_p(file_mapping, 'mapping of stimuli')
    # Output
    if SHOW_OUTPUT:
        analysis = tr.analysis.Analysis()
        num_stimuli = tr.common.get_configs('num_stimuli')
        logger.info('Creating figures.')
        # Visualisation of keypress data
        if SHOW_OUTPUT_KP:
            # all keypresses with confidence interval
            analysis.plot_kp(mapping, conf_interval=0.95)
            # # keypresses of all individual stimuli
            # logger.info('Creating figures for keypress data of individual stimuli.')
            # for stim in tqdm(range(num_stimuli)):  # tqdm adds progress bar
            #     # extract timestamps of events
            #     vert_lines = list(map(int, re.findall(r'\d+', mapping.loc['video_' + str(stim), 'events'])))
            #     # convert to s
            #     vert_lines = [x / 1000 for x in vert_lines]  # type: ignore
            #     # extract annotations
            #     vert_line_annotations = mapping.loc['video_' + str(stim), 'events_description'].split(',')
            #     # remove [
            #     vert_line_annotations[0] = vert_line_annotations[0][1:]
            #     # remove ]
            #     vert_line_annotations[-1] = vert_line_annotations[-1][:-1]
            #     # plot
            #     analysis.plot_kp_video(mapping,
            #                            'video_' + str(stim),
            #                            vert_lines=vert_lines,
            #                            vert_lines_width=1,
            #                            vert_lines_dash='solid',
            #                            vert_lines_colour='red',
            #                            vert_lines_annotations=vert_line_annotations,
            #                            vert_lines_annotations_position='top right',
            #                            vert_lines_annotations_font_size=12,
            #                            vert_lines_annotations_colour='red',
            #                            conf_interval=0.95)
            # keypresses of groups of stimuli
            logger.info('Creating plots of keypress data for groups of stimuli.')
            for stim in tqdm(range(int(num_stimuli/4))):  # tqdm adds progress bar
                # ids of stimuli that belong to the same group
                ids = [stim, stim + int(num_stimuli/4), stim + int(num_stimuli/4*2), stim + int(num_stimuli/4*3)]
                df = mapping[mapping['id'].isin(ids)]
                # extract timestamps of events
                events = []
                vert_lines = list(map(int, re.findall(r'\d+', df.loc['video_' + str(stim), 'events'])))
                vert_lines_ids = list(map(int, re.findall(r'\d+', df.loc['video_' + str(stim), 'events_id'])))
                # convert to s
                vert_lines = [x / 1000 for x in vert_lines]  # type: ignore
                # extract annotations
                vert_line_annotations = df.loc['video_' + str(stim), 'events_name'].split(',')
                # remove [
                vert_line_annotations[0] = vert_line_annotations[0][1:]
                # remove ]
                vert_line_annotations[-1] = vert_line_annotations[-1][:-1]
                # add info to dictionary of events to be passed for plotting
                for x in range(0, len(vert_line_annotations)):
                    # search for start and end values
                    start_found = False  # toggle for finding the start x coordinate
                    start = 0  # x coordinate of starting location of event
                    end = 0  # x coordinate of ending location of event
                    # search for start and end x coordinates of event
                    for y in range(0, len(vert_lines_ids)):
                        # check if start is at the same location
                        if vert_lines_ids[y] == x + 1 and not start_found:
                            start = vert_lines[y]
                            start_found = True
                        # check if end is at the same location
                        elif vert_lines_ids[y] == x + 1 and start_found:
                            end = vert_lines[y]
                    # add to dictionary of events
                    events.append({'id': x + 1,
                                   'start': start,
                                   'end': end,
                                   'annotation': vert_line_annotations[x]})
                # prepare pairs of signals to compare with ttest
                # ttest_signals = []  # list of dictionaries
                # todo: @Shadab, create list of things to compare using ttest here
                # 0 and 1 = within (paired): https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_rel.html  # noqa: E501
                # 0 and 2 = between: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html
                # 0 and 3 = between
                # 1 and 2 = between
                # 2 and 3 = within
                # 1 and 3 = between
                # todo: @Shadab, we should find some clever way to understand which once should be between and which
                #       one within. Code below could then be used
                # for stim in ids[1:]:
                #     # assume that df[ids[0]] is always the baseline to compare against
                #     ttest_signals.append({'signal_1': df.loc['video_' + str(ids[0])]['kp'],
                #                           'signal_2': df.loc['video_' + str(stim)]['kp'],
                #                           'paired': True})
                ttest_signals = [{'signal_1': df.loc['video_' + str(ids[0])]['kp_raw'][0],  # 0 and 1 = within
                                  'signal_2': df.loc['video_' + str(ids[1])]['kp_raw'][0],
                                  'label': 'ttest(0, 1)',
                                  'paired': True},
                                 {'signal_1': df.loc['video_' + str(ids[0])]['kp_raw'][0],  # 0 and 2 = between
                                  'signal_2': df.loc['video_' + str(ids[2])]['kp_raw'][0],
                                  'label': 'ttest(0, 2)',
                                  'paired': False},
                                 {'signal_1': df.loc['video_' + str(ids[0])]['kp_raw'][0],  # 0 and 3 = between
                                  'signal_2': df.loc['video_' + str(ids[3])]['kp_raw'][0],
                                  'label': 'ttest(0, 3)',
                                  'paired': False},
                                 {'signal_1': df.loc['video_' + str(ids[1])]['kp_raw'][0],  # 1 and 2 = between
                                  'signal_2': df.loc['video_' + str(ids[2])]['kp_raw'][0],
                                  'label': 'ttest(1, 2)',
                                  'paired': False},
                                 {'signal_1': df.loc['video_' + str(ids[2])]['kp_raw'][0],  # 2 and 3 = within
                                  'signal_2': df.loc['video_' + str(ids[3])]['kp_raw'][0],
                                  'label': 'ttest(2, 3)',
                                  'paired': True},
                                 {'signal_1': df.loc['video_' + str(ids[1])]['kp_raw'][0],  # 1 and 3 = between
                                  'signal_2': df.loc['video_' + str(ids[3])]['kp_raw'][0],
                                  'label': 'ttest(1, 3)',
                                  'paired': False}]
                # signal_1 = signal_type = list of int, eg: [1,1,0,0]
                # signal_2 = signal_ego = list of int, eg: [1,1,0,0]
                # signal_3 = signal_kp = list of lists, eg: [[1,1,1,1], [1,1,1,1], [1,1,1,1], [1,1,1,1]]
                # prepare signals to compare with ANOVA
                # todo: signals for ANOVA
                anova_signals = [{'signal_1': df.loc['video_' + str(ids[0])]['kp'],
                                  'signal_2': df.loc['video_' + str(ids[0])]['kp'],
                                  'signal_3': df.loc['video_' + str(ids[0])]['kp'],
                                  'label': 'anova(0, 1, 2)'},
                                 {'signal_1': df.loc['video_' + str(ids[0])]['kp'],
                                  'signal_2': df.loc['video_' + str(ids[0])]['kp'],
                                  'signal_3': df.loc['video_' + str(ids[0])]['kp'],
                                  'label': 'anova(0, 2, 3)'},
                                 {'signal_1': df.loc['video_' + str(ids[0])]['kp'],
                                  'signal_2': df.loc['video_' + str(ids[0])]['kp'],
                                  'signal_3': df.loc['video_' + str(ids[0])]['kp'],
                                  'label': 'anova(1, 2, 3)'}]
                # plot keypress data and slider questions
                analysis.plot_kp_slider_videos(df,
                                               y=['comfort', 'safety', 'expectation'],
                                               # hardcode based on the longest stimulus
                                               xaxis_kp_range=[0, 43],
                                               # hardcode based on the highest recorded value with space for ttest and
                                               # anova markers
                                               yaxis_kp_range=[0, 65],
                                               events=events,
                                               events_width=1,
                                               events_dash='dot',
                                               events_colour='black' if tr.common.get_configs('plotly_template') == 'plotly_white' else 'white',  # noqa: E501
                                               events_annotations_font_size=12,
                                               events_annotations_colour='black' if tr.common.get_configs('plotly_template') == 'plotly_white' else 'white',  # noqa: E501
                                               yaxis_slider_title=None,
                                               show_text_labels=True,
                                               stacked=True,
                                               yaxis_slider_show=False,
                                               font_size=16,
                                               legend_x=0.68,
                                               legend_y=1.0,
                                               fig_save_width=1600,   # preserve ratio 225x152
                                               fig_save_height=1080,  # preserve ratio 225x152
                                               name_file='kp_videos_sliders_'+','.join([str(i) for i in ids]),
                                               ttest_signals=ttest_signals,
                                               ttest_marker='circle',
                                               ttest_marker_size=3,
                                               ttest_marker_colour='black',
                                               ttest_annotations_font_size=10,
                                               ttest_annotations_colour='black',
                                               anova_signals=anova_signals,
                                               anova_marker='cross',
                                               anova_marker_size=3,
                                               anova_marker_colour='black',
                                               anova_annotations_font_size=10,
                                               anova_annotations_colour='black',)
            # keypresses of an individual stimulus for an individual pp
            # analysis.plot_kp_video_pp(mapping,
            #                           heroku_data,
            #                           pp='R51701197342646JF16777X',
            #                           stimulus='video_2',
            #                           conf_interval=0.95)
            # keypresses of all videos individually
            analysis.plot_kp_videos(mapping, show_menu=False, show_title=False)
            # keypress based on the type of ego car
            # todo: double check that order of AV/MDV is correct
            analysis.plot_kp_variable(mapping,
                                      'ego_car',
                                      y_legend=['AV', 'MDV'],
                                      font_size=16,
                                      legend_x=0.9,
                                      legend_y=1.0,
                                      show_menu=False,
                                      show_title=False)
            # keypress based on the type of ego car
            # todo: double check that order of AV/MDV is correct
            analysis.plot_kp_variable(mapping,
                                      'target_car',
                                      y_legend=['AV', 'MDV'],
                                      font_size=16,
                                      legend_x=0.9,
                                      legend_y=1.0,
                                      show_menu=False,
                                      show_title=False)
            # keypress based on the pp group
            analysis.plot_kp_variable(mapping,
                                      'group',
                                      # custom labels for slider questions in the legend
                                      y_legend=['Group 1', 'Group 2', 'Group 3', 'Group 4'],
                                      font_size=16,
                                      legend_x=0.9,
                                      legend_y=1.0,
                                      show_menu=False,
                                      show_title=False)
        # Visualisation of stimulus data
        if SHOW_OUTPUT_ST:
            # post stimulus questions for all stimuli
            analysis.bar(mapping,
                         y=['comfort', 'safety', 'expectation'],
                         stacked=True,
                         show_text_labels=True,
                         pretty_text=True,
                         save_file=True)
            # # post-trial questions of all groups of stimuli
            # logger.info('Creating bar plots of post-trial questions for groups of stimuli.')
            # for stim in tqdm(range(int(num_stimuli/4))):  # tqdm adds progress bar
            #     # get ids of stimuli that belong to the same group
            #     ids = [stim, stim + int(num_stimuli/4), stim + int(num_stimuli/4*2), stim + int(num_stimuli/4*3)]
            #     df = mapping[mapping['id'].isin(ids)]
            #     analysis.bar(df,
            #                  y=['comfort', 'safety', 'expectation'],
            #                  stacked=True,
            #                  show_text_labels=True,
            #                  pretty_text=True,
            #                  save_file=True)
            # columns to drop in correlation matrix and scatter matrix
            columns_drop = ['id', 'description', 'video_length', 'min_dur', 'max_dur', 'kp', 'kp_raw', 'events',
                            'events_name', 'events_description', 'events_id', 'description', 'video_name']
            # set nan to -1
            df = mapping.fillna(-1)
            # create correlation matrix
            analysis.corr_matrix(df,
                                 columns_drop=columns_drop,
                                 save_file=True)
            # create correlation matrix
            analysis.scatter_matrix(df,
                                    columns_drop=columns_drop,
                                    color='group',
                                    symbol='group',
                                    diagonal_visible=False,
                                    save_file=True)
            # participant group - end question
            analysis.scatter(heroku_data,
                             x='participant_group',
                             y='end-slider-0-0',
                             color='end-slider-1-0',
                             pretty_text=True,
                             save_file=True)
            # stimulus duration
            analysis.hist(heroku_data,
                          x=heroku_data.columns[heroku_data.columns.to_series().str.contains('-dur')],
                          nbins=100,
                          pretty_text=True,
                          save_file=True)
            # browser window dimensions
            analysis.scatter(heroku_data,
                             x='window_width',
                             y='window_height',
                             color='browser_name',
                             pretty_text=True,
                             save_file=True)
            # mapping to convert likert values to numeric
            likert_mapping = {'Strongly disagree': 1,
                              'Disagree': 2,
                              'Neither disagree nor agree': 3,
                              'Agree': 4,
                              'Strongly agree': 5}
            # questions before and after
            df = all_data
            df['driving_alongside_ad'] = df['driving_alongside_ad'].map(likert_mapping)
            df['driving_in_ad'] = df['driving_in_ad'].map(likert_mapping)
            analysis.scatter(df,
                             x='driving_alongside_ad',
                             y='end-slider-0-0',
                             xaxis_title='Before',
                             yaxis_title='After',
                             pretty_text=False,
                             save_file=True)
            analysis.scatter(df,
                             x='driving_in_ad',
                             y='end-slider-1-0',
                             xaxis_title='Before',
                             yaxis_title='After',
                             pretty_text=False,
                             save_file=True)
        # Visualisation of data about participants
        if SHOW_OUTPUT_PP:
            # time of participation
            df = appen_data
            df['country'] = df['country'].fillna('NaN')
            df['time'] = df['time'] / 60.0  # convert to min
            # histogram of duration of participation
            analysis.hist(df,
                          x=['time'],
                          color='country',
                          pretty_text=True,
                          save_file=True)
            # histogram of driving frequency
            analysis.hist(appen_data,
                          x=['driving_freq'],
                          pretty_text=True,
                          save_file=True)
            # grouped barchart of DBQ data
            analysis.hist(appen_data,
                          x=['dbq1_anger',
                             'dbq2_speed_motorway',
                             'dbq3_speed_residential',
                             'dbq4_headway',
                             'dbq5_traffic_lights',
                             'dbq6_horn',
                             'dbq7_mobile'],
                          marginal='violin',
                          pretty_text=True,
                          save_file=True)
            # histogram of the year of license
            analysis.hist(appen_data,
                          x=['year_license'],
                          pretty_text=True,
                          save_file=True)
            # histogram of the highest level of education
            analysis.hist(appen_data,
                          x=['education'],
                          pretty_text=True,
                          save_file=True)
            # histogram of communication with other
            analysis.hist(appen_data,
                          x=['communication_others'],
                          marginal='violin',
                          pretty_text=True,
                          save_file=True)
            # grouped barchart of technology scale
            analysis.hist(appen_data,
                          x=['technology_worried',
                             'technology_enjoyment',
                             'technology_lives_easier',
                             'technology_lives_change',
                             'technology_not_interested'],
                          marginal='violin',
                          pretty_text=True,
                          save_file=True)
            # scatter plot of machines questions
            analysis.scatter(appen_data,
                             x='machines_roles',
                             y='machines_profit',
                             color='year_license',
                             pretty_text=True,
                             save_file=True)
            # histogram of attitude towards AD
            analysis.hist(appen_data,
                          x=['attitude_ad'],
                          pretty_text=True,
                          save_file=True)
            # scatter plot of driving with AVs
            analysis.scatter(appen_data,
                             x='driving_in_ad',
                             y='driving_alongside_ad',
                             color='year_license',
                             pretty_text=True,
                             save_file=True)
            # histogram of the capability of AD
            analysis.hist(appen_data,
                          x=['capability_ad'],
                          pretty_text=True,
                          save_file=True)
            # histogram of the experience of AD
            analysis.hist(appen_data,
                          x=['experience_ad'],
                          pretty_text=True,
                          save_file=True)
            # histogram of the input device
            analysis.hist(appen_data,
                          x=['device'],
                          pretty_text=True,
                          save_file=True)
            # histogram of milage
            analysis.hist(appen_data,
                          x=['milage'],
                          pretty_text=True,
                          save_file=True)
            # map of participants
            analysis.map(countries_data, color='counts', save_file=True)
            # map of mean age per country
            analysis.map(countries_data, color='age', save_file=True)
            # map of gender per country
            analysis.map(countries_data, color='gender', save_file=True)
            # map of year of obtaining license per country
            analysis.map(countries_data, color='year_license', save_file=True)
            # map of year of automated driving per country
            analysis.map(countries_data, color='year_ad', save_file=True)
        # Visualisation of eye tracking data
        if SHOW_OUTPUT_ET:
            # create eye gaze visualisations for all videos
            logger.info('Producing visualisations of eye gaze data for {} stimuli.',
                        tr.common.get_configs('num_stimuli'))
            if tr.common.get_configs('combined_animation') == 1:
                num_anim = 21
                logger.info('Animation is set to combined animations of all for scenarios in one figure')
            else:
                num_anim = tr.common.get_configs('num_stimuli')
                logger.info('Animation is set to single stimuli animations in one figure')
            # source video/stimulus for a given individual.
            for id_video in tqdm(range(0, num_anim)):
                logger.info('Producing visualisations of eye gaze data for stimulus {}.', id_video)
                # Deconstruct the source video into its individual frames.
                stim_path = os.path.join(tr.settings.output_dir, 'frames')
                # To allow for overlaying the heatmap for each frame later on.
                analysis.save_all_frames(heroku_data, mapping, id_video=id_video, t='video_length')
                # create animation for stimulus
                points_process = {}
                points_process1 = {}
                points_process2 = {}
                points_process3 = {}
                # determin amount of points in duration for video_id
                dur = mapping.iloc[id_video]['video_length']
                hm_resolution_range = int(50000 / tr.common.get_configs('hm_resolution'))
                # To create animation for scenario 1,2,3 & 4 in the
                # same animation extract for all senarios.
                # for individual animations or scenario
                dur = heroku_data['video_'+str(id_video)+'-dur-0'].tolist()
                dur = [x for x in dur if str(x) != 'nan']
                dur = int(round(mean(dur) / 1000) * 1000)
                hm_resolution_range = int(50000 / tr.common.get_configs('hm_resolution'))
                # for individual stim
                for points_dur in range(0, hm_resolution_range, 1):
                    try:
                        points_process[points_dur] = points_duration[points_dur][id_video]
                    except KeyError:
                        break
                # check if animations is set for combined
                if tr.common.get_configs('combined_animation') == 1:
                    # Scenario 2
                    for points_dur in range(0, hm_resolution_range, 1):
                        try:
                            points_process1[points_dur] = points_duration[points_dur][id_video + 21]
                        except KeyError:
                            break
                    # Scenario 3
                    for points_dur in range(0, hm_resolution_range, 1):
                        try:
                            points_process2[points_dur] = points_duration[points_dur][id_video + 42]
                        except KeyError:
                            break
                    # Scenario 4
                    for points_dur in range(0, hm_resolution_range, 1):
                        try:
                            points_process3[points_dur] = points_duration[points_dur][id_video + 63]
                        except KeyError:
                            break
                analysis.create_animation(heroku_data,
                                          mapping,
                                          stim_path,
                                          id_video,
                                          points_process,
                                          points_process1,
                                          points_process2,
                                          points_process3,
                                          t='video_length',
                                          save_anim=True,
                                          save_frames=True)
                # stitch animations into 1 long videos
                analysis.create_animation_all_stimuli(num_stimuli)
        # collect figure objects
        figures = [manager.canvas.figure
                   for manager in
                   matplotlib._pylab_helpers.Gcf.get_all_fig_managers()]
        # show figures, if any
        if figures:
            plt.show()
