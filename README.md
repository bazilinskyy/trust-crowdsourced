# Analysing trust in a traffic scene with an automated vehicle

This project defines a framework for the analysis of the level of trust in a traffic environment involving an automated vehicle. The jsPsych framework is used to for the frontend. In the description below, it is assumed that the repo is stored in the folder `trust-crowdsourced`. Terminal commands lower assume macOS.

## Setup
Tested with Python 3.9.12. To setup the environment run these two commands in a parent folder of the downloaded repository (replace `/` with `\` and possibly add `--user` if on Windows):
- `pip install -e trust-crowdsourced` will setup the project as a package accessible in the environment.
- `pip install -r trust-crowdsourced/requirements.txt` will install required packages.

### Configuration of project
Configuration of the project needs to be defined in `trust-crowdsourced/config`. Please use the `default.config` file for the required structure of the file. If no custom config file is provided, `default.config` is used. The config file has the following parameters:
* `appen_job`: ID of the appen job.
* `num_stimuli`: number of stimuli in the study.
* `num_stimuli_participant`: subset of stimuli in the study shown to each participant.
* `allowed_min_time`: the cut-off for minimal time of participation for filtering.
* `num_repeat`: number of times each stimulus is repeated.
* `kp_resolution`: bin size in ms in which data is stored.
* `allowed_stimulus_wrong_duration`: if the percentage of videos with abnormal length is above this value, exclude participant from analysis.
* `allowed_mistakes_signs`: number of allowed mistakes in the questions about traffic signs.
* `sign_answers`: answers to the questions on traffic signs.
* `mask_id`: number for masking worker IDs in appen data.
* `files_heroku`: files with data from heroku.
* `files_appen`: files with data from appen.
* `file_cheaters`: CSV file with cheaters for flagging.
* `path_source`: path with source files for the stimuli from the Unity3D project.
* `path_stimuli`: path consisting of all videos included in the survey.
* `mapping_stimuli`: CSV file that contains all data found in the videos.
* `plotly_template`: template used to make graphs in the analysis.
* `stimulus_width`: width of stimuli.
* `stimulus_height`: height of stimuli.
* `aoi`: csv file with AOI data.
* `separate_lab`: separate data from the lab experiment and crowdsourced experiments.
* `smoothen_signal`: toggle to apply filter to smoothen data.
* `freq`: frequency used by One Euro Filter.
* `mincutoff`: minimal cutoff used by One Euro Filter.
* `beta`: beta value used by One Euro Filter.
* `dcutoff`: d-cutoff value used by One Euro Filter.
* `font_family`: font family to be used on the figures.
* `font_size`: font size to be used on the figures.
* `p_value`: p value used for ttest.
* `save_figures`: save "final" figures to the /figures folder.

## Preparation of stimuli
The source files of the video stimuli are outputted from Unity to `config.path_source`. To prepare them for the crowdsourced setup `python trust-crowdsourced/preparation/process_videos.py`. Videos will be outputted to `config.path_stimuli`.

## Analysis
Analysis can be started by running python `trust-crowdsourced/trust/run.py`. A number of CSV files used for data processing are saved in `trust-crowdsourced/_output`. Visualisations of all data are saved in `trust-crowdsourced/_output/figures/`.

## Keypress data
### All participants

[![plot_all_all_videos](figures/kp_videos.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/kp_videos.html)
Percentage of participants pressing the response key as a function of elapsed video time for all stimuli for all participants.

[![plot_all_group](figures/kp_group-0-1-2-3.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/kp_group-0-1-2-3.html)
Percentage of participants pressing the response key as a function of elapsed video time for groups of scenarios for all participants.

[![plot_all_ego](figures/kp_ego_car-0-1.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/kp_ego_car-0-1.html)
Percentage of participants pressing the response key as a function of elapsed video time for two types of ego car for all participants.

[![plot_all_target](figures/kp_target_car-0-1.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/kp_target_car-0-1.html)
Percentage of participants pressing the response key as a function of elapsed video time for two types of target car for all participants.

[![plot_all_0](figures/kp_videos_sliders_0,21,42,63.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/kp_videos_sliders_0,21,42,63.html)
Percentage of participants pressing the response key as a function of elapsed video time and responses to post-stimulus questions for scenario 0 (baseline) for all participants.

[![plot_all_1](figures/kp_videos_sliders_1,22,43,64.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/kp_videos_sliders_1,22,43,64.html)
Percentage of participants pressing the response key as a function of elapsed video time and responses to post-stimulus questions for scenario 1 for all participants.

[![plot_all_2](figures/kp_videos_sliders,2,23,44,65.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/kp_videos_sliders_2,23,44,65.html)
Percentage of participants pressing the response key as a function of elapsed video time and responses to post-stimulus questions for scenario 2 for all participants.

[![plot_all_3](figures/kp_videos_sliders_3,24,45,66.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/kp_videos_sliders_3,24,45,66.html)
Percentage of participants pressing the response key as a function of elapsed video time and responses to post-stimulus questions for scenario 3 for all participants.

[![plot_all_4](figures/kp_videos_sliders_4,25,46,67.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/kp_videos_sliders_4,25,46,67.html)
Percentage of participants pressing the response key as a function of elapsed video time and responses to post-stimulus questions for scenario 4 for all participants.

[![plot_all_5](figures/kp_videos_sliders_5,26,47,68.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/kp_videos_sliders_5,26,47,68.html)
Percentage of participants pressing the response key as a function of elapsed video time and responses to post-stimulus questions for scenario 5 for all participants.

[![plot_all_6](figures/kp_videos_sliders_6,27,48,69.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/kp_videos_sliders_6,27,48,69.html)
Percentage of participants pressing the response key as a function of elapsed video time and responses to post-stimulus questions for scenario 6 for all participants.

[![plot_all_7](figures/kp_videos_sliders_7,28,49,70.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/kp_videos_sliders_7,28,49,70.html)
Percentage of participants pressing the response key as a function of elapsed video time and responses to post-stimulus questions for scenario 7 for all participants.

[![plot_all_8](figures/kp_videos_sliders_8,29,50,71.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/kp_videos_sliders_8,29,50,71.html)
Percentage of participants pressing the response key as a function of elapsed video time and responses to post-stimulus questions for scenario 8 for all participants.

[![plot_all_9](figures/kp_videos_sliders_9,30,51,72.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/kp_videos_sliders_9,30,51,72.html)
Percentage of participants pressing the response key as a function of elapsed video time and responses to post-stimulus questions for scenario 9 for all participants.

[![plot_all_10](figures/kp_videos_sliders_10,31,52,73.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/kp_videos_sliders_10,31,52,73.html)
Percentage of participants pressing the response key as a function of elapsed video time and responses to post-stimulus questions for scenario 10 for all participants.

[![plot_all_11](figures/kp_videos_sliders_11,32,53,74.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/kp_videos_sliders_11,32,53,74.html)
Percentage of participants pressing the response key as a function of elapsed video time and responses to post-stimulus questions for scenario 11 for all participants.

[![plot_all_12](figures/kp_videos_sliders_12,33,54,75.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/kp_videos_sliders_12,33,54,75.html)
Percentage of participants pressing the response key as a function of elapsed video time and responses to post-stimulus questions for scenario 12 for all participants.

[![plot_all_13](figures/kp_videos_sliders_13,34,55,76.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/kp_videos_sliders_13,34,55,76.html)
Percentage of participants pressing the response key as a function of elapsed video time and responses to post-stimulus questions for scenario 13 for all participants.

[![plot_all_14](figures/kp_videos_sliders_14,35,56,77.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/kp_videos_sliders_14,35,56,77.html)
Percentage of participants pressing the response key as a function of elapsed video time and responses to post-stimulus questions for scenario 14 for all participants.

[![plot_all_15](figures/kp_videos_sliders_15,36,57,78.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/kp_videos_sliders_15,36,57,78.html)
Percentage of participants pressing the response key as a function of elapsed video time and responses to post-stimulus questions for scenario 15 for all participants.

[![plot_all_16](figures/kp_videos_sliders_16,37,58,79.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/kp_videos_sliders_16,37,58,79.html)
Percentage of participants pressing the response key as a function of elapsed video time and responses to post-stimulus questions for scenario 16 for all participants.

[![plot_all_17](figures/kp_videos_sliders_17,38,59,80.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/kp_videos_sliders_17,38,59,80.html)
Percentage of participants pressing the response key as a function of elapsed video time and responses to post-stimulus questions for scenario 17 for all participants.

[![plot_all_18](figures/kp_videos_sliders_18,39,60,81.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/kp_videos_sliders_18,39,60,81.html)
Percentage of participants pressing the response key as a function of elapsed video time and responses to post-stimulus questions for scenario 18 for all participants.

[![plot_all_19](figures/kp_videos_sliders_19,40,61,82.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/kp_videos_sliders_19,40,61,82.html)
Percentage of participants pressing the response key as a function of elapsed video time and responses to post-stimulus questions for scenario 19 for all participants.

[![plot_all_20](figures/kp_videos_sliders_20,41,62,83.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/kp_videos_sliders_20,41,62,83.html)
Percentage of participants pressing the response key as a function of elapsed video time and responses to post-stimulus questions for scenario 20 for all participants.

### For only lab participants
todo

#### Correlation and scatter matrices
![correlation matrix](https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/corr_matrix.jpg?raw=true)  
Correlation matrix.

[![scatter matrix](figures/scatter_matrix.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/scatter_matrix.html)  
Scatter matrix.

## Area of Interest (AOI)
### For all participants
[![plot_all_0](figures/AOI_0.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/AOI_0.html)
Plot of AOI analysis for video 0 for all participants.

[![plot_all_1](figures/AOI_1.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/AOI_1.html)
Plot of AOI analysis for video 1 for all participants.

[![plot_all_2](figures/AOI_2.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/AOI_2.html)
Plot of AOI analysis for video 2 for all participants.

[![plot_all_3](figures/AOI_3.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/AOI_3.html)
Plot of AOI analysis for video 3 for all participants.

[![plot_all_4](figures/AOI_4.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/AOI_4.html)
Plot of AOI analysis for video 4 for all participants.

[![plot_all_5](figures/AOI_5.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/AOI_5.html)
Plot of AOI analysis for video 5 for all participants.

[![plot_all_6](figures/AOI_6.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/AOI_6.html)
Plot of AOI analysis for video 6 for all participants.

[![plot_all_7](figures/AOI_7.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/AOI_7.html)
Plot of AOI analysis for video 7 for all participants.

[![plot_all_8](figures/AOI_8.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/AOI_8.html)
Plot of AOI analysis for video 8 for all participants.

[![plot_all_9](figures/AOI_9.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/AOI_9.html)
Plot of AOI analysis for video 9 for all participants.

[![plot_all_10](figures/AOI_10.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/AOI_10.html)
Plot of AOI analysis for video 10 for all participants.

[![plot_all_11](figures/AOI_11.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/AOI_11.html)
Plot of AOI analysis for video 11 for all participants.

[![plot_all_12](figures/AOI_12.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/AOI_12.html)
Plot of AOI analysis for video 12 for all participants.

[![plot_all_13](figures/AOI_13.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/AOI_13.html)
Plot of AOI analysis for video 13 for all participants.

[![plot_all_14](figures/AOI_14.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/AOI_14.html)
Plot of AOI analysis for video 14 for all participants.

[![plot_all_15](figures/AOI_15.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/AOI_15.html)
Plot of AOI analysis for video 15 for all participants.

[![plot_all_16](figures/AOI_16.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/AOI_16.html)
Plot of AOI analysis for video 16 for all participants.

[![plot_all_17](figures/AOI_17.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/AOI_17.html)
Plot of AOI analysis for video 17 for all participants.

[![plot_all_18](figures/AOI_18.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/AOI_18.html)
Plot of AOI analysis for video 18 for all participants.

[![plot_all_19](figures/AOI_19.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/AOI_19.html)
Plot of AOI analysis for video 19 for all participants.

[![plot_all_20](figures/AOI_20.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/aoi_20.html)
Plot of AOI analysis for video 20 for all participants.

### For only lab participants
[![plot_lab_only_0](figures/Lab_only_AOI_0.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/Lab_only_AOI_0.html)
Plot of AOI analysis for video 0 for lab participants.

[![plot_lab_only_1](figures/Lab_only_AOI_1.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/Lab_only_AOI_1.html)
Plot of AOI analysis for video 1 for lab participants.

[![plot_lab_only_2](figures/Lab_only_AOI_2.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/Lab_only_AOI_2.html)
Plot of AOI analysis for video 2 for lab participants.

[![plot_lab_only_3](figures/Lab_only_AOI_3.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/Lab_only_AOI_3.html)
Plot of AOI analysis for video 3 for lab participants.

[![plot_lab_only_4](figures/Lab_only_AOI_4.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/Lab_only_AOI_4.html)
Plot of AOI analysis for video 4 for lab participants.

[![plot_lab_only_5](figures/Lab_only_AOI_5.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/Lab_only_AOI_5.html)
Plot of AOI analysis for video 5 for lab participants.

[![plot_lab_only_6](figures/Lab_only_AOI_6.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/Lab_only_AOI_6.html)
Plot of AOI analysis for video 6 for lab participants.

[![plot_lab_only_7](figures/Lab_only_AOI_7.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/Lab_only_AOI_7.html)
Plot of AOI analysis for video 7 for lab participants.

[![plot_lab_only_8](figures/Lab_only_AOI_8.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/Lab_only_AOI_8.html)
Plot of AOI analysis for video 8 for lab participants.

[![plot_lab_only_9](figures/Lab_only_AOI_9.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/Lab_only_AOI_9.html)
Plot of AOI analysis for video 9 for lab participants.

[![plot_lab_only_10](figures/Lab_only_AOI_10.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/Lab_only_AOI_10.html)
Plot of AOI analysis for video 10 for lab participants.

[![plot_lab_only_11](figures/Lab_only_AOI_11.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/Lab_only_AOI_11.html)
Plot of AOI analysis for video 11 for lab participants.

[![plot_lab_only_12](figures/Lab_only_AOI_12.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/Lab_only_AOI_12.html)
Plot of AOI analysis for video 12 for lab participants.

[![plot_lab_only_13](figures/Lab_only_AOI_13.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/Lab_only_AOI_13.html)
Plot of AOI analysis for video 13 for lab participants.

[![plot_lab_only_14](figures/Lab_only_AOI_14.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/Lab_only_AOI_14.html)
Plot of AOI analysis for video 14 for lab participants.

[![plot_lab_only_15](figures/Lab_only_AOI_15.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/Lab_only_AOI_15.html)
Plot of AOI analysis for video 15 for lab participants.

[![plot_lab_only_16](figures/Lab_only_AOI_16.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/Lab_only_AOI_16.html)
Plot of AOI analysis for video 16 for lab participants.

[![plot_lab_only_17](figures/Lab_only_AOI_17.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/Lab_only_AOI_17.html)
Plot of AOI analysis for video 17 for lab participants.

[![plot_lab_only_18](figures/Lab_only_AOI_18.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/Lab_only_AOI_18.html)
Plot of AOI analysis for video 18 for lab participants.

[![plot_lab_only_19](figures/Lab_only_AOI_19.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/Lab_only_AOI_19.html)
Plot of AOI analysis for video 19 for lab participants.

[![plot_lab_only_20](figures/Lab_only_AOI_20.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/Lab_only_AOI_20.html)
Plot of AOI analysis for video 20 for lab participants.

#### Information on participants
[![driving frequency](figures/hist_driving_freq.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/hist_driving_freq.html)  
Driving frequency.

[![mileage](figures/hist_milage.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/hist_milage.html)  
Mileage.

[![input device](figures/hist_device.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/hist_device.html)  
Input device.

[![driving behaviour questionnaire](figures/hist_dbq1_anger-dbq2_speed_motorway-dbq3_speed_residential-dbq4_headway-dbq5_traffic_lights-dbq6_horn-dbq7_mobile.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/hist_dbq1_anger-dbq2_speed_motorway-dbq3_speed_residential-dbq4_headway-dbq5_traffic_lights-dbq6_horn-dbq7_mobile.html)  
Driving behaviour questionnaire (DBQ).

[![time of participation](figures/hist_time.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/hist_time.html)  
Time of participation.

[![year of license](figures/hist_year_license.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/hist_year_license.html)  
Year of obtaining driver's license.

[![education](figures/hist_education.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/hist_education.html)  
Highest obtained level of education.

[![communication_others](figures/hist_communication_others.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/hist_communication_others.html)  
Responses to statement "I would like to communicate with other road users while driving (for instance, using eye contact, gestures, verbal communication, etc.)".

[![technology](figures/hist_technology_worried-technology_enjoyment-technology_lives_easier-technology_lives_change-technology_not_interested.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/hist_technology_worried-technology_enjoyment-technology_lives_easier-technology_lives_change-technology_not_interested.html)  
Technology acceptance scale.

[![machines](figures/scatter_machines_roles-machines_profit.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/scatter_machines_roles-machines_profit.html)  
Responses to x:"I enjoy making use of the latest technological products and services when I have the opportunity" and y:"New technologies are all about making profits rather than making people's lives better".

[![attitude AD](figures/hist_attitude_ad.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/hist_attitude_ad.html)  
Responses to statement "Please indicate your general attitude towards automated cars".

[![driving with AD](figures/scatter_driving_in_ad-driving_alongside_ad.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/scatter_driving_in_ad-driving_alongside_ad.html)  
Responses to x:"When the autonomous vehicle is on the road, I would feel comfortable about driving on roads alongside autonomous cars" and y:"When the autonomous vehicle is on the road, I would feel comfortable about
using an autonomous car instead of driving a traditional car.".

[![capability of AD](figures/hist_capability_ad.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/hist_capability_ad.html)  
Responses to question "Who do you think is more capable of conducting driving-related tasks?"

[![experience of AD](figures/hist_experience_ad.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/hist_experience_ad.html)  
Responses to question "Which options best describes your experience with automated cars?"

[![map of counts of participants](figures/map_counts.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/map_counts.html)  
Map of counts of participants.

[![map of years of having a license](figures/map_year_license.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/map_year_license.html)  
Map of years of having a license.

[![map of prediction of year of introduction of automated cars](figures/map_year_ad.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/map_year_ad.html)  
Map of prediction of the year of introduction of automated cars in the country of residence.

[![map of age](figures/map_age.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/map_age.html)  
Map of age of participants.

[![map of gender](figures/map_gender.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/map_gender.html)  
Map of distribution of gender.

#### Technical characteristics of participants
[![dimensions of browser](figures/scatter_window_width-window_height.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/scatter_window_width-window_height.html)  
Dimensions of browser.

## Troubleshooting
### Troubleshooting setup
#### ERROR: trust-crowdsourced is not a valid editable requirement
Check that you are indeed in the parent folder for running command `pip install -e trust-crowdsourced`. This command will not work from inside of the folder containing the repo.