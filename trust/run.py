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
# SHOW_OUTPUT_KP = True  # should figures with keypress data be plotted-
# SHOW_OUTPUT_ST = True  # should figures with stimulus data to be plotted
# SHOW_OUTPUT_PP = True  # should figures with info about participants
# SHOW_OUTPUT_ET = False  # should figures for eye tracking

# for debugging, skip processing
SAVE_P = False  # save pickle files with data
LOAD_P = True  # load pickle files with data
SAVE_CSV = True  # load csv files with data
FILTER_DATA = True  # filter Appen and heroku data
CLEAN_DATA = True  # clean Appen data
REJECT_CHEATERS = False  # reject cheaters on Appen
CALC_COORDS = False  # extract points from heroku data
UPDATE_MAPPING = True  # update mapping with keypress data
SHOW_OUTPUT = True  # should figures be plotted
SHOW_OUTPUT_KP = True  # should figures with keypress data be plotted
SHOW_OUTPUT_ST = True  # should figures with stimulus data be plotted
SHOW_OUTPUT_PP = True  # should figures with info about participants be plotted
SHOW_OUTPUT_ET = False  # should figures for eye tracking be plotted
USE_ONEEURO_FILTER = True
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

    # Directly count participants in each group
    if 'participant_group' in heroku_data.columns:
        group_counts = heroku_data['participant_group'].value_counts()
        print("Participant Counts by Group:")
        for group, count in group_counts.items():
            print(f"Group {group}: {count} participants")
    else:
        print("'participant_group' column not found in the data.")
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
        questions = [{'question': 'slider-0',
                      'type': 'num'},
                     {'question': 'slider-1',
                      'type': 'num'},
                     {'question': 'slider-2',
                      'type': 'num'}]
        # process post-trial questions and update mapping
        mapping = heroku.process_stimulus_questions(questions)
        # export to pickle
        tr.common.save_to_p(file_mapping,
                            mapping,
                            'mapping of stimuli')
    else:
        mapping = tr.common.load_from_p(file_mapping, 'mapping of stimuli')
    # Output
    if SHOW_OUTPUT:
        analysis = tr.analysis.Analysis()
        num_stimuli = tr.common.get_configs('num_stimuli')
        logger.info('Creating figures.')
        # Visualisation of keypress data
        if SHOW_OUTPUT_KP:
            # # all keypresses with confidence interval
            # analysis.plot_kp(mapping, conf_interval=0.95)
            # # keypresses of all individual stimuli
            # logger.info(
            #'Creating figures for keypress data of individual stimuli.')
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
            logger.info('Creating bar plots of keypress data for groups of stimuli.')
            for stim in tqdm(range(int(num_stimuli/4))):  # tqdm adds progress bar
                # ids of stimuli that belong to the same group
                ids = [stim, stim + int(num_stimuli/4), stim + int(num_stimuli/4*2), stim + int(num_stimuli/4*3)]
                df = mapping[mapping['id'].isin(ids)]
                # extract timestamps of events
                vert_lines = list(map(int, re.findall(r'\d+', df.loc['video_' + str(stim), 'events'])))
                # convert to s
                vert_lines = [x / 1000 for x in vert_lines]  # type: ignore
                # extract annotations
                vert_line_annotations = df.loc['video_' + str(stim), 'events_description'].split(',')
                # remove [
                vert_line_annotations[0] = vert_line_annotations[0][1:]
                # remove ]
                vert_line_annotations[-1] = vert_line_annotations[-1][:-1]
                # create list with ids of labels for outputting just numbers
                vert_line_annotations_num = []
                for x in range(1, len(vert_line_annotations) + 1):
                    vert_line_annotations_num.append(x)
                # # plot keypresses for each video€
                # analysis.plot_kp_videos(df,
                #                         vert_lines=vert_lines,
                #                         vert_lines_width=1,
                #                         vert_lines_dash='solid',
                #                         vert_lines_colour='red',
                #                         vert_lines_annotations=vert_line_annotations,
                #                         vert_lines_annotations_position='top right',
                #                         vert_lines_annotations_font_size=12,
                #                         vert_lines_annotations_colour='red',
                #                         name_file='kp_videos_'+','.join([str(i) for i in ids]))
                # plot keypress data and slider questions
                analysis.plot_kp_slider_videos(df,
                                               y=['slider-0', 'slider-1', 'slider-2'],
                                               xaxis_kp_range=[0, 43],  # hardcode based on the longest stimulus
                                               yaxis_kp_range=[0, 60],  # hardcode based on the highest recorded value
                                               vert_lines=vert_lines,
                                               vert_lines_width=1,
                                               vert_lines_dash='solid',
                                               vert_lines_colour='black',
                                               vert_lines_annotations=vert_line_annotations_num,
                                               vert_lines_annotations_position='top right',
                                               vert_lines_annotations_font_size=12,
                                               vert_lines_annotations_colour='black',
                                               yaxis_slider_title=None,
                                               show_text_labels=True,
                                               stacked=True,
                                               yaxis_slider_show=False,
                                               name_file='kp_videos_sliders_'+','.join([str(i) for i in ids]),
                                               use_one_euro_filter=USE_ONEEURO_FILTER  # Pass the global flag
                                            )
            # # keypresses of an individual stimulus for an individual pp
            # analysis.plot_kp_video_pp(mapping,
            #                           heroku_data,
            #                           pp='R51701197342646JF16777X',
            #                           stimulus='video_2',
            #                           conf_interval=0.95,
            #                           use_one_euro_filter=USE_ONEEURO_FILTER  # Pass the global flag
            #                       )
            # keypresses of all videos individually
            analysis.plot_kp_videos(mapping, show_menu=False, use_one_euro_filter=False)
            # keypress based on the type of ego car
            analysis.plot_kp_variable(mapping, 'ego_car', show_menu=False, use_one_euro_filter=False)
            # keypress based on the type of ego car
            analysis.plot_kp_variable(mapping, 'target_car', show_menu=False, use_one_euro_filter=False)
            # keypress based on the pp group
            analysis.plot_kp_variable(mapping, 'group', show_menu=False, use_one_euro_filter=False)
        # Visualisation of stimulus data
        if SHOW_OUTPUT_ST:
            # post stimulus questions for all stimuli
            analysis.bar(mapping,
                         y=['slider-0', 'slider-1', 'slider-2'],
                         stacked=True,
                         show_text_labels=True,
                         pretty_text=True,
                         save_file=True)
            # post-trial questions of all individual stimuli
            logger.info('Creating bar plots of post-trial questions for groups of stimuli.')
            for stim in tqdm(range(int(num_stimuli/4))):  # tqdm adds progress bar
                # get ids of stimuli that belong to the same group
                ids = [stim, stim + int(num_stimuli/4), stim + int(num_stimuli/4*2), stim + int(num_stimuli/4*3)]
                df = mapping[mapping['id'].isin(ids)]
                analysis.bar(df,
                             y=['slider-0', 'slider-1', 'slider-2'],
                             stacked=True,
                             show_text_labels=True,
                             pretty_text=True,
                             save_file=True)
            # columns to drop in correlation matrix and scatter matrix
            columns_drop = ['events_description', 'description', 'video_length', 'min_dur', 'max_dur', 'kp', 'events']
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
            analysis.hist(df,
                          x=['time'],
                          color='country',
                          pretty_text=True,
                          save_file=True)
            # driving with AVs
            analysis.scatter(appen_data,
                             x='driving_in_ad',
                             y='driving_alongside_ad',
                             color='year_license',
                             pretty_text=True,
                             save_file=True)
            # histogram for driving frequency
            analysis.hist(appen_data,
                          x=['driving_freq'],
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
                logger.info('Producing visualisations of eye gaze data for stimulus {}.',
                            id_video)
                # Deconstruct the source video into its individual frames.
                stim_path = os.path.join(tr.settings.output_dir, 'frames')
                # To allow for overlaying the heatmap for each frame later on.
                analysis.save_all_frames(heroku_data,
                                         mapping,
                                         id_video=id_video,
                                         t='video_length'
                                         )
                # create animation for stimulus
                points_process = {}
                points_process1 = {}
                points_process2 = {}
                points_process3 = {}
                # determin amount of points in duration for video_id
                dur = mapping.iloc[id_video]['video_length']
                hm_resolution_range = int(50000/tr.common.get_configs('hm_resolution'))
                # To create animation for scenario 1,2,3 & 4 in the
                # same animation extract for all senarios.
                # for individual animations or scenario
                dur = heroku_data['video_'+str(id_video)+'-dur-0'].tolist()
                dur = [x for x in dur if str(x) != 'nan']
                dur = int(round(mean(dur)/1000)*1000)
                hm_resolution_range = int(50000/tr.common.get_configs('hm_resolution'))
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
                            points_process1[points_dur] = points_duration[points_dur][id_video+21]
                        except KeyError:
                            break
                    # Scenario 3
                    for points_dur in range(0, hm_resolution_range, 1):
                        try:
                            points_process2[points_dur] = points_duration[points_dur][id_video+42]
                        except KeyError:
                            break
                    # Scenario 4
                    for points_dur in range(0, hm_resolution_range, 1):
                        try:
                            points_process3[points_dur] = points_duration[points_dur][id_video+63]
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

        figures = [manager.canvas.figure
                   for manager in
                   matplotlib._pylab_helpers.Gcf.get_all_fig_managers()]
        # show figures, if any
        if figures:
            plt.show()
