# by Pavlo Bazilinskyy <pavlo.bazilinskyy@gmail.com>
import matplotlib.pyplot as plt
import matplotlib._pylab_helpers
from tqdm import tqdm
import os
# import shutil
import trust as tr
from statistics import mean

tr.logs(show_level='info', show_color=True)
logger = tr.CustomLogger(__name__)  # use custom logger

# const
# SAVE_P = True  # save pickle files with data
# LOAD_P = False  # load pickle files with data
# SAVE_CSV = True  # load csv files with data
# FILTER_DATA = True  # filter Appen and heroku data
# CLEAN_DATA = True  # clean Appen data
# REJECT_CHEATERS = True  # reject cheaters on Appen
# CALC_COORDS = False
# UPDATE_MAPPING = True  # update mapping with keypress data
# SHOW_OUTPUT = True  # should figures be plotted
# SHOW_OUTPUT_KP = True  # should figures with keypress data be plotted
# SHOW_OUTPUT_ST = True  # should figures with stimulus data to be plotted
# SHOW_OUTPUT_PP = True  # should figures with info about participants
# SHOW_OUTPUT_ET = True  # should figures for eye tracking

# for debugging, skip processing
SAVE_P = False  # save pickle files with data
LOAD_P = True  # load pickle files with data
SAVE_CSV = True  # load csv files with data
FILTER_DATA = False  # filter Appen and heroku data
CLEAN_DATA = True  # clean Appen data
REJECT_CHEATERS = False  # reject cheaters on Appen
CALC_COORDS = True  # extract points from heroku data
UPDATE_MAPPING = True  # update mapping with keypress data
SHOW_OUTPUT = True  # should figures be plotted
SHOW_OUTPUT_KP = False  # should figures with keypress data be plotted
SHOW_OUTPUT_ST = False  # should figures with stimulus data to be plotted
SHOW_OUTPUT_PP = False  # should figures with info about participants
SHOW_OUTPUT_ET = True  # should figures for eye tracking

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
        qa = tr.analysis.QA(file_cheaters=tr.common.get_configs('file_cheaters'),  # noqa: E501
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
    heroku_data = all_data[all_data.columns.intersection(heroku_data_keys)]
    heroku_data = heroku_data.set_index('worker_code')
    heroku.set_data(heroku_data)  # update object with filtered data
    appen_data = all_data[all_data.columns.intersection(appen_data_keys)]
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
        mapping = tr.common.load_from_p(file_mapping,
                                        'mapping of stimuli')
    if SHOW_OUTPUT:
        # Output
        analysis = tr.analysis.Analysis()
        num_stimuli = tr.common.get_configs('num_stimuli')
        logger.info('Creating figures.')
        # Visualisation of keypress data
        if SHOW_OUTPUT_KP:
            # all keypresses with confidence interval
            analysis.plot_kp(mapping, conf_interval=0.95)
            # keypresses of an individual stimulus
            # analysis.plot_kp_video(mapping, 'video_0', conf_interval=0.95)
            # keypresses of an individual stimulus for an individual pp
            analysis.plot_kp_video_pp(mapping,
                                      heroku_data,
                                      pp='R51701197342646JF16777X',
                                      stimulus='video_2',
                                      conf_interval=0.95)
            # keypresses of all videos individually
            analysis.plot_kp_videos(mapping)
            # 1 var, all values
            analysis.plot_kp_variable(mapping, 'ego_car')
            # 1 var, certain values
            analysis.plot_kp_variable(mapping,
                                      'target_car',
                                      [0, 1])
            analysis.plot_kp_variable(mapping,
                                      'group',
                                      [0, 1, 2, 3])
            # TODO: make plot_video_data work
            # plot of multiple combined AND variables
            analysis.plot_video_data(mapping, 'video_5',
                                     ['group', 'criticality'],
                                     yaxis_title='Type of ego car and number of pedestrians',  # noqa: E501
                                     conf_interval=0.95)
            analysis.plot_kp_variables_and(mapping,
                                           plot_names=['traffic rules',
                                                       'no traffic rules'],
                                           variables_list=[[{'variable': 'traffic_rules',  # noqa: E501
                                                             'value': 'stop_sign'},        # noqa: E501
                                                            {'variable': 'traffic_rules',  # noqa: E501
                                                             'value': 'traffic_lights'},   # noqa: E501
                                                            {'variable': 'traffic_rules',  # noqa: E501
                                                             'value': 'ped_crossing'}],    # noqa: E501
                                                           [{'variable': 'traffic_rules',  # noqa: E501
                                                             'value': 'none'}]])  # noqa: E501
            # plot of separate variables
            analysis.plot_kp_variables_or(mapping,
                                          variables=[{'variable': 'cross_look',  # noqa: E501
                                                      'value': 'Crossing_Looking'},     # noqa: E501
                                                     {'variable': 'cross_look',  # noqa: E501
                                                      'value': 'notCrossing_Looking'},  # noqa: E501
                                                     {'variable': 'cross_look',  # noqa: E501
                                                      'value': 'Crossing_notLooking'},  # noqa: E501
                                                     {'variable': 'cross_look',  # noqa: E501
                                                      'value': 'nonspecific'}])  # noqa: E501
        # Visualisation of stimulus data
        if SHOW_OUTPUT_ST:
            # post-trial questions
            # note: post-stimulus slider questions are stored as video_0-as-0 in  # noqa: E501
            #       the form of eg [100, 0, 0].
            analysis.bar(heroku_data,
                         y=['video_0-slider-0-0', 'video_0-slider-1-0', 'video_0-slider-2-0'],  # noqa: E501
                         pretty_text=True,
                         save_file=True)
            analysis.hist(heroku_data,
                          x=heroku_data.columns[heroku_data.columns.to_series().str.contains('-slider-')],  # noqa: E501
                          nbins=100,
                          pretty_text=True,
                          save_file=True)
            # columns to drop in correlation matrix and scatter matrix
            columns_drop = ['description', 'video_length', 'min_dur',
                            'max_dur', 'kp']
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
                          x=heroku_data.columns[heroku_data.columns.to_series().str.contains('-dur')],  # noqa: E501
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
            df['driving_alongside_ad'] = df['driving_alongside_ad'].map(likert_mapping)  # noqa: E501
            df['driving_in_ad'] = df['driving_in_ad'].map(likert_mapping)
            analysis.scatter(df,
                             x='driving_alongside_ad',  # noqa: E501
                             y='end-slider-0-0',  # noqa: E501
                             xaxis_title='Before',
                             yaxis_title='After',
                             pretty_text=True,
                             save_file=True)
            analysis.scatter(df,
                             x='driving_in_ad',  # noqa: E501
                             y='end-slider-1-0',  # noqa: E501
                             xaxis_title='Before',
                             yaxis_title='After',
                             pretty_text=True,
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
        if SHOW_OUTPUT_ET:            # df = all_data
            # create eye gaze visualisations for all videos
            logger.info('Producing visualisations of eye gaze data for {} stimuli.',  # noqa: E501
                        tr.common.get_configs('num_stimuli'))
            # stimulus videos with manual ego and target car
            video_0_0 = range(0, 20, 1)
            # stimulus vidoe with manual ego car but av target car
            video_0_1 = range(21, 41, 1)
            # stimulus video with av ego car but manual target car
            video_1_0 = range(42, 62, 1)
            # stimulus video with av ego and target car
            video_1_1 = range(63, 83, 1)

            # source video/stimulus for a given individual.
            for id_video in tqdm(range(0, tr.common.get_configs(
                                       'num_stimuli')-1)):
                logger.info('Producing visualisations of eye gaze data for stimulus {}.',  # noqa: E501
                            id_video)
                # Deconstruct the source video into its individual frames.
                stim_path = os.path.join(tr.settings.output_dir, 'frames')
                # To allow for overlaying the heatmap for each frame later on.
                analysis.save_all_frames(heroku_data,
                                         mapping,
                                         id_video=id_video,
                                         t='video_length'
                                         )
                # construct the gazes lines just as an example for how
                # that looks compared to the heatmap.

                # analysis.create_gazes(heroku_data,
                #                       x='video_'+str(id_video)+'-x-0',
                #                       y='video_'+str(id_video)+'-y-0',

                #                       # pp='R51701252541887JF46247X',
                #                       id_video=id_video,
                #                       save_file=True)
                # Construct heatmap over each video frame previously created
                # from the source video.
                # create histogram for stimulus
                # analysis.create_histogram(stim_path,
                #                   points[id_video],
                #                   id_video=id_video,
                #                   density_coef=20,
                #                   save_file=True)
                # # create animation for stimulus
                points_process = {}
                # determin amount of points in duration for video_id
<<<<<<< Updated upstream
                dur = heroku_data['video_'+str(id_video)+'-dur-0'].tolist()
                dur = [x for x in dur if str(x) != 'nan']
                dur = int(round(mean(dur)/1000)*1000)
=======
                dur = mapping.iloc[id_video]['video_length']
                hm_resolution_range = int(50000/tr.common.get_configs('hm_resolution'))  # noqa: E501
>>>>>>> Stashed changes
                # for individual
                for points_dur in range(0, len(range(0, dur,
                                               tr.common.get_configs(
                                                   'hm_resolution')))):
                    points_process[points_dur] = points_duration[points_dur][id_video]  # noqa: E501
                analysis.create_animation(heroku_data,
                                          mapping,
                                          stim_path,
                                          id_video,
                                          points_process,
                                          t='video_length',
                                          save_anim=True,
                                          save_frames=True)
                # analysis.create_heatmap(heroku_data,
                #                         x='video_'+str(id_video)+'-x-0',
                #                         y='video_'+str(id_video)+'-y-0',
                #                         pp='R51701252541887JF46247X',
                #                         id_video=id_video,
                #                         type_heatmap='contourf',
                #                         add_corners=True,
                #                         save_file=True)
                # # Animate the kp for given source video.
                # analysis.plot_kp_animate(mapping,
                #                          'video_'+str(id_video),
                #                          conf_interval=0.95)
                #
                # # Create an animation from individual frames
                # #
                # analysis.create_animation(heroku_data,
                #                           x='video_'+str(id_video)+'-x-0',
                #                           y='video_'+str(id_video)+'-y-0',
                #                           t='video_'+str(id_video)+'-t-0',
                #                           pp='R51701252541887JF46247X',
                #                           id_video=id_video,
                #                           save_anim=True,
                #                           save_frames=True)
                # # remove temp folder with frames
                # shutil.rmtree(os.path.join(tr.settings.output_dir, 'frames'))
            # todo: add comment with description
            analysis.scatter_mult(heroku_data,
                                  x=['video_0-x-0', 'video_1-x-0'],
                                  y='video_0-y-0',
                                  color='browser_major_version',
                                  pretty_text=True,
                                  save_file=True)
            # Creating a for loop that makes an eye gazes/heatmap for every
            # create animation for stimulus
            # analysis.scatter_mult(mapping[mapping['avg_person'] != ''],     # noqa: E501
            #                       x=['avg_object', 'avg_person', 'avg_car'],
            #                       y='avg_kp',
            #                       trendline='ols',
            #                       xaxis_title='Object count',
            #                       yaxis_title='Mean keypresses (%)',
            #                       marginal_y=None,
            #                       marginal_x='rug',
            #                       save_file=True)
            # Create individual scatter plot for given video and participant.
            analysis.scatter_et(heroku_data,
                                x='video_0-x-0',
                                y='video_0-y-0',
                                t='video_0-t-0',
                                pp='R51701252541887JF46247X',
                                id_video='video_0',
                                pretty_text=True,
                                save_file=True)
            # Create individual heatmap for given video and participant.
            # analysis.heatmap(heroku_data,
            #                     x='video_0-x-0',
            #                     y='video_0-y-0',
            #                     t='video_0-t-0',
            #                     pp='R51701252541887JF46247X',
            #                     id_video='video_0',
            #                     pretty_text=True,
            #                     save_file=True)
        # stitch animations into 1 long videos
        analysis.create_animation_all_stimuli(num_stimuli)
        # all keypresses with confidence interval
        analysis.plot_kp(mapping, conf_interval=0.95)
        # keypresses of an individual stimulus
        analysis.plot_kp_video(mapping,
                               pp='R51701270114366JF82237X',
                               stimulus='video_0',
                               conf_interval=0.95)
        # keypresses of an individual stimulus for an individual pp
        analysis.plot_kp_video(mapping,
                               pp='R51701270114366JF82237X',
                               stimulus='video_0',
                               conf_interval=0.95)
        # keypresses of all videos individually
        analysis.plot_kp_videos(mapping)
        # 1 var, all values
        analysis.plot_kp_variable(mapping, 'ego_car',
                                  [0, 1])
        # 1 var, certain values
        analysis.plot_kp_variable(mapping,
                                  'target_car',
                                  [0, 1])
        analysis.plot_kp_variable(mapping,
                                  'group',
                                  [0, 1, 2, 3])
        # TODO: make plot_video_data work
        # plot of multiple combined AND variables
        # analysis.plot_video_data(mapping, 'video_5',
        #                          ['group', 'criticality'],
        #                          yaxis_title='Type of ego car and number of pedestrians',  # noqa: E501
        #                          conf_interval=0.95)
        # analysis.plot_kp_variables_and(mapping,
        #                                plot_names=['traffic rules',
        #                                            'no traffic rules'],
        #                                variables_list=[[{'variable': 'traffic_rules',  # noqa: E501
        #                                                  'value': 'stop_sign'},        # noqa: E501
        #                                                 {'variable': 'traffic_rules',  # noqa: E501
        #                                                  'value': 'traffic_lights'},   # noqa: E501
        #                                                 {'variable': 'traffic_rules',  # noqa: E501
        #                                                  'value': 'ped_crossing'}],    # noqa: E501
        #                                                [{'variable': 'traffic_rules',  # noqa: E501
        #                                                  'value': 'none'}]])
        # # plot of separate variables
        # analysis.plot_kp_variables_or(mapping,
        #                               variables=[{'variable': 'cross_look',
        #                                           'value': 'Crossing_Looking'},     # noqa: E501
        #                                          {'variable': 'cross_look',
        #                                           'value': 'notCrossing_Looking'},  # noqa: E501
        #                                          {'variable': 'cross_look',
        #                                           'value': 'Crossing_notLooking'},  # noqa: E501
        #                                          {'variable': 'cross_look',
        #                                           'value': 'nonspecific'}])
        # post-trial questions
        # analysis.bar(heroku_data,
        #              y=['video_0-slider-0-0', 'video_0-slider-1-0', 'video_0-slider-2-0'],  # noqa: E501
        #              pretty_text=True,
        #              save_file=True)

        analysis.scatter_mult(heroku_data,
                              x=['video_0-x-0', 'video_1-x-0'],
                              y='video_0-y-0',
                              color='browser_major_version',
                              pretty_text=True,
                              save_file=True)

        figures = [manager.canvas.figure
                   for manager in
                   matplotlib._pylab_helpers.Gcf.get_all_fig_managers()]
        # show figures, if any
        if figures:
            plt.show()
