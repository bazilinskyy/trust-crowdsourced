# by Pavlo Bazilinskyy <pavlo.bazilinskyy@gmail.com>
import matplotlib.pyplot as plt
import matplotlib._pylab_helpers
from tqdm import tqdm
import os
import shutil
import trust as tr

tr.logs(show_level='info', show_color=True)
logger = tr.CustomLogger(__name__)  # use custom logger

# const
# SAVE_P = True  # save pickle files with data
# LOAD_P = False  # load pickle files with data
# SAVE_CSV = True  # load csv files with data
# FILTER_DATA = True  # filter Appen and heroku data
# CLEAN_DATA = True  # clean Appen data
# REJECT_CHEATERS = True  # reject cheaters on Appen
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
CLEAN_DATA = False  # clean Appen data
REJECT_CHEATERS = False  # reject cheaters on Appen
UPDATE_MAPPING = False  # update mapping with keypress data
SHOW_OUTPUT = True  # should figures be plotted
SHOW_OUTPUT_KP = False  # should figures with keypress data be plotted
SHOW_OUTPUT_ST = False  # should figures with stimulus data to be plotted
SHOW_OUTPUT_PP = False  # should figures with info about participants
SHOW_OUTPUT_ET = True  # should figures for eye tracking

file_mapping = 'mapping.p'  # file to save updated mapping

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
            print(df.head)
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
        if SHOW_OUTPUT_ET:
            # create eye gaze visualisations for all videos
            logger.info('Producing visualisations of eye gaze data for {} stimuli.',  # noqa: E501
                        tr.common.get_configs('num_stimuli'))
            # source video/stimulus for a given individual.
            for id_video in tqdm(range(tr.common.get_configs('num_stimuli'))):
                logger.info('Producing visualisations of eye gaze data for stimulus {}.',  # noqa: E501
                            id_video)
                # Deconstruct the source video into its individual frames.
                # To allow for overlaying the heatmap for each frame later on.
                analysis.save_all_frames(heroku_data,
                                         id_video=id_video,
                                         t='video_'+str(id_video)+'-t-0',
                                         id_pp=6)
                # construct the gazes lines just as an example for how
                # that looks compared to the heatmap.
                # todo: rename id_pp to pp everywhere
                # todo: use worker_code instead of numeric ID pp='R51701197342646JF16777X'  # noqa: E501
                analysis.create_gazes(heroku_data,
                                      x='video_'+str(id_video)+'-x-0',
                                      y='video_'+str(id_video)+'-y-0',
                                      id_pp=6,
                                      id_video=id_video,
                                      save_file=True)
                # Construct heatmap over each video frame previously created
                # from the source video.
                analysis.create_heatmap(heroku_data,
                                        x='video_'+str(id_video)+'-x-0',
                                        y='video_'+str(id_video)+'-y-0',
                                        id_pp=6,
                                        id_video=id_video,
                                        type_heatmap='contourf',
                                        add_corners=True,
                                        save_file=True)
                # Animate the kp for given source video.
                analysis.plot_kp_animate(mapping,
                                         'video_'+str(id_video),
                                         conf_interval=0.95)
                # todo: @Job, add comment and what method below does
                analysis.create_animation(heroku_data,
                                          x='video_'+str(id_video)+'-x-0',
                                          y='video_'+str(id_video)+'-y-0',
                                          t='video_'+str(id_video)+'-t-0',
                                          id_pp=6,
                                          id_video=id_video,
                                          save_anim=True,
                                          save_frames=True)
                # remove temp folder with frames
                shutil.rmtree(os.path.join(tr.settings.output_dir, 'frames'))
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
                                id_pp=6,
                                id_video='video_0',
                                pretty_text=True,
                                save_file=True)
            # Create individual heatmap for given video and participant.
            # analysis.heatmap(heroku_data,
            #                     x='video_0-x-0',
            #                     y='video_0-y-0',
            #                     t='video_0-t-0',
            #                     id_pp=6,
            #                     id_video='video_0',
            #                     pretty_text=True,
            #                     save_file=True)

        # check if any figures are to be rendered
        figures = [manager.canvas.figure
                   for manager in
                   matplotlib._pylab_helpers.Gcf.get_all_fig_managers()]
        # show figures, if any
        if figures:
            plt.show()
