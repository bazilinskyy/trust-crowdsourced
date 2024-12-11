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
* `file_appen`: file with data from appen.
* `file_cheaters`: CSV file with cheaters for flagging.
* `path_source`: path with source files for the stimuli from the Unity3D project.
* `path_stimuli`: path consisting of all videos included in the survey.
* `mapping_stimuli`: CSV file that contains all data found in the videos.
* `plotly_template`: template used to make graphs in the analysis.

## Preparation of stimuli
The source files of the video stimuli are outputted from Unity to `config.path_source`. To prepare them for the crowdsourced setup `python trust-crowdsourced/preparation/process_videos.py`. Videos will be outputted to `config.path_stimuli`.

## Troubleshooting
### Troubleshooting setup
#### ERROR: trust-crowdsourced is not a valid editable requirement
Check that you are indeed in the parent folder for running command `pip install -e trust-crowdsourced`. This command will not work from inside of the folder containing the repo.

## Figures
For the analysis plots of the AOI data were made for two groups. 
## Area of Interest (AOI)
### For all participants

[![plot_all_0](figures/aoi_0.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/aoi_0.html)
Plot of AOI analysis for video 0 for all participants.

[![plot_all_1](figures/aoi_1.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/aoi_1.html)
Plot of AOI analysis for video 1 for all participants.

[![plot_all_2](figures/aoi_2.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/aoi_2.html)
Plot of AOI analysis for video 2 for all participants.

[![plot_all_3](figures/aoi_3.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/aoi_3.html)
Plot of AOI analysis for video 3 for all participants.

[![plot_all_4](figures/aoi_4.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/aoi_4.html)
Plot of AOI analysis for video 4 for all participants.

[![plot_all_5](figures/aoi_5.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/aoi_5.html)
Plot of AOI analysis for video 5 for all participants.

[![plot_all_6](figures/aoi_6.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/aoi_6.html)
Plot of AOI analysis for video 6 for all participants.

[![plot_all_7](figures/aoi_7.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/aoi_7.html)
Plot of AOI analysis for video 7 for all participants.

[![plot_all_8](figures/aoi_8.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/aoi_8.html)
Plot of AOI analysis for video 8 for all participants.

[![plot_all_9](figures/aoi_9.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/aoi_9.html)
Plot of AOI analysis for video 9 for all participants.

[![plot_all_10](figures/aoi_10.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/aoi_10.html)
Plot of AOI analysis for video 10 for all participants.

[![plot_all_11](figures/aoi_11.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/aoi_11.html)
Plot of AOI analysis for video 11 for all participants.

[![plot_all_12](figures/aoi_12.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/aoi_12.html)
Plot of AOI analysis for video 12 for all participants.

[![plot_all_13](figures/aoi_13.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/aoi_13.html)
Plot of AOI analysis for video 13 for all participants.

[![plot_all_14](figures/aoi_14.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/aoi_14.html)
Plot of AOI analysis for video 14 for all participants.

[![plot_all_15](figures/aoi_15.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/aoi_15.html)
Plot of AOI analysis for video 15 for all participants.

[![plot_all_16](figures/aoi_16.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/aoi_16.html)
Plot of AOI analysis for video 16 for all participants.

[![plot_all_17](figures/aoi_17.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/aoi_17.html)
Plot of AOI analysis for video 17 for all participants.

[![plot_all_18](figures/aoi_18.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/aoi_18.html)
Plot of AOI analysis for video 18 for all participants.

[![plot_all_19](figures/aoi_19.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/aoi_19.html)
Plot of AOI analysis for video 19 for all participants.

[![plot_all_20](figures/aoi_20.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/aoi_20.html)
Plot of AOI analysis for video 20 for all participants.

### For only lab participants

[![plot_lab_only_0](figures/lab_only_aoi_0.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/lab_only_aoi_0.html)
Plot of AOI analysis for video 0 for lab participants.

[![plot_lab_only_1](figures/lab_only_aoi_1.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/lab_only_aoi_1.html)
Plot of AOI analysis for video 1 for lab participants.

[![plot_lab_only_2](figures/lab_only_aoi_2.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/lab_only_aoi_2.html)
Plot of AOI analysis for video 2 for lab participants.

[![plot_lab_only_3](figures/lab_only_aoi_3.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/lab_only_aoi_3.html)
Plot of AOI analysis for video 3 for lab participants.

[![plot_lab_only_4](figures/lab_only_aoi_4.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/lab_only_aoi_4.html)
Plot of AOI analysis for video 4 for lab participants.

[![plot_lab_only_5](figures/lab_only_aoi_5.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/lab_only_aoi_5.html)
Plot of AOI analysis for video 5 for lab participants.

[![plot_lab_only_6](figures/lab_only_aoi_6.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/lab_only_aoi_6.html)
Plot of AOI analysis for video 6 for lab participants.

[![plot_lab_only_7](figures/lab_only_aoi_7.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/lab_only_aoi_7.html)
Plot of AOI analysis for video 7 for lab participants.

[![plot_lab_only_8](figures/lab_only_aoi_8.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/lab_only_aoi_8.html)
Plot of AOI analysis for video 8 for lab participants.

[![plot_lab_only_9](figures/lab_only_aoi_9.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/lab_only_aoi_9.html)
Plot of AOI analysis for video 9 for lab participants.

[![plot_lab_only_10](figures/lab_only_aoi_10.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/lab_only_aoi_10.html)
Plot of AOI analysis for video 10 for lab participants.

[![plot_lab_only_11](figures/lab_only_aoi_11.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/lab_only_aoi_11.html)
Plot of AOI analysis for video 11 for lab participants.

[![plot_lab_only_12](figures/lab_only_aoi_12.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/lab_only_aoi_12.html)
Plot of AOI analysis for video 12 for lab participants.

[![plot_lab_only_13](figures/lab_only_aoi_13.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/lab_only_aoi_13.html)
Plot of AOI analysis for video 13 for lab participants.

[![plot_lab_only_14](figures/lab_only_aoi_14.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/lab_only_aoi_14.html)
Plot of AOI analysis for video 14 for lab participants.

[![plot_lab_only_15](figures/lab_only_aoi_15.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/lab_only_aoi_15.html)
Plot of AOI analysis for video 15 for lab participants.

[![plot_lab_only_16](figures/lab_only_aoi_16.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/lab_only_aoi_16.html)
Plot of AOI analysis for video 16 for lab participants.

[![plot_lab_only_17](figures/lab_only_aoi_17.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/lab_only_aoi_17.html)
Plot of AOI analysis for video 17 for lab participants.

[![plot_lab_only_18](figures/lab_only_aoi_18.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/lab_only_aoi_18.html)
Plot of AOI analysis for video 18 for lab participants.

[![plot_lab_only_19](figures/lab_only_aoi_19.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/lab_only_aoi_19.html)
Plot of AOI analysis for video 19 for lab participants.

[![plot_lab_only_20](figures/lab_only_aoi_20.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/lab_only_aoi_20.html)
Plot of AOI analysis for video 20 for lab participants.

## Keypress analysis
### All participants
[![plot_all_0](figures/kp_videos_sliders_0,21,42,63.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/smooth-kp-one-euro-filter/figures/kp_videos_sliders_0,21,42,63.html)
Plot of keypress analysis and slider data for videos 0, 21, 42, 63 for all participants.

[![plot_all_1](figures/kp_videos_sliders_1,22,43,64.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/smooth-kp-one-euro-filter/figures/kp_videos_sliders_1,22,43,64.html)
Plot of keypress analysis and slider data for videos 1, 22, 43, 64 for all participants.

[![plot_all_2](figures/kp_videos_sliders,2,23,44,65.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/smooth-kp-one-euro-filter/figures/kp_videos_sliders_2,23,44,65.html)
Plot of keypress analysis and slider data of video 2 for all participants.

[![plot_all_3](figures/kp_videos_sliders_3,24,45,66.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/smooth-kp-one-euro-filter/figures/kp_videos_sliders_3,24,45,66.html)
Plot of keypress analysis and slider data of video 3 for all participants.

[![plot_all_4](figures/kp_videos_sliders_4,25,46,67.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/smooth-kp-one-euro-filter/figures/kp_videos_sliders_4,25,46,67.html)
Plot of keypress analysis and slider data of video 4 for all participants.

[![plot_all_5](figures/kp_videos_sliders_5,26,47,68.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/smooth-kp-one-euro-filter/figures/kp_videos_sliders_5,26,47,68.html)
Plot of keypress analysis and slider data of video 5 for all participants.

[![plot_all_6](figures/kp_videos_sliders_6,27,48,69.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/smooth-kp-one-euro-filter/figures/kp_videos_sliders_6,27,48,69.html)
Plot of keypress analysis and slider data of video 6 for all participants.

[![plot_all_7](figures/kp_videos_sliders_7,28,49,70.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/smooth-kp-one-euro-filter/figures/kp_videos_sliders_7,28,49,70.html)
Plot of keypress analysis and slider data of video 7 for all participants.

[![plot_all_8](figures/kp_videos_sliders_8,29,50,71.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/smooth-kp-one-euro-filter/figures/kp_videos_sliders_8,29,50,71.html)
Plot of keypress analysis and slider data of video 8 for all participants.

[![plot_all_9](figures/kp_videos_sliders_9,30,51,72.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/smooth-kp-one-euro-filter/figures/kp_videos_sliders_9,30,51,72.html)
Plot of keypress analysis and slider data of video 9 for all participants.

[![plot_all_10](figures/kp_videos_sliders_10,31,52,73.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/smooth-kp-one-euro-filter/figures/kp_videos_sliders_10,31,52,73.html)
Plot of keypress analysis and slider data of video 10 for all participants.

[![plot_all_11](figures/kp_videos_sliders_11,32,53,74.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/smooth-kp-one-euro-filter/figures/kp_videos_sliders_11,32,53,74.html)
Plot of keypress analysis and slider data of video 11 for all participants.

[![plot_all_12](figures/kp_videos_sliders_12,33,54,75.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/smooth-kp-one-euro-filter/figures/kp_videos_sliders_12,33,54,75.html)
Plot of keypress analysis and slider data of video 12 for all participants.

[![plot_all_13](figures/kp_videos_sliders_13,34,55,76.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/smooth-kp-one-euro-filter/figures/kp_videos_sliders_13,34,55,76.html)
Plot of keypress analysis and slider data of video 13 for all participants.

[![plot_all_14](figures/kp_videos_sliders_14,35,56,77.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/smooth-kp-one-euro-filter/figures/kp_videos_sliders_14,35,56,77.html)
Plot of keypress analysis and slider data of video 14 for all participants.

[![plot_all_15](figures/kp_videos_sliders_15,36,57,78.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/smooth-kp-one-euro-filter/figures/kp_videos_sliders_15,36,57,78.html)
Plot of keypress analysis and slider data of video 15 for all participants.

[![plot_all_16](figures/kp_videos_sliders_16,37,58,79.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/smooth-kp-one-euro-filter/figures/kp_videos_sliders_16,37,58,79.html)
Plot of keypress analysis and slider data of video 16 for all participants.

[![plot_all_17](figures/kp_videos_sliders_17,38,59,80.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/smooth-kp-one-euro-filter/figures/kp_videos_sliders_17,38,59,80.html)
Plot of keypress analysis and slider data of video 17 for all participants.

[![plot_all_18](figures/kp_videos_sliders_18,39,60,81.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/smooth-kp-one-euro-filter/figures/kp_videos_sliders_18,39,60,81.html)
Plot of keypress analysis and slider data of video 18 for all participants.

[![plot_all_19](figures/kp_videos_sliders_19,40,61,82.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/smooth-kp-one-euro-filter/figures/kp_videos_sliders_19,40,61,82.html)
Plot of keypress analysis and slider data of video 19 for all participants.

[![plot_all_20](figures/kp_videos_sliders_20,41,62,83.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/smooth-kp-one-euro-filter/figures/kp_videos_sliders_20,41,62,83.html)
Plot of keypress analysis and slider data of video 20 for all participants.

### For only lab participants
todo

#### Information on participants
[![driving frequency](figures/hist_driving_freq.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/crossing-crowdsourcing/blob/main/figures/hist_driving_freq.html)  
Driving frequency.

[![driving behaviour questionnaire](figures/hist_dbq.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/crossing-crowdsourcing/blob/main/figures/hist_dbq.html)  
Driving behaviour questionnaire (DBQ).

[![time of participation](figures/hist_time.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/crossing-crowdsourcing/blob/main/figures/hist_time.html)  
Time of participation.

[![map of counts of participants](figures/map_counts.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/crossing-crowdsourcing/blob/main/figures/map_counts.html)  
Map of counts of participants.

[![map of years of having a license](figures/map_year_license.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/crossing-crowdsourcing/blob/main/figures/map_year_license.html)  
Map of years of having a license.

[![map of prediction of year of introduction of automated cars](figures/map_year_ad.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/crossing-crowdsourcing/blob/main/figures/map_year_ad.html)  
Map of prediction of the year of introduction of automated cars in the country of residence.

[![map of age](figures/map_age.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/crossing-crowdsourcing/blob/main/figures/map_age.html)  
Map of age of participants.

[![map of gender](figures/map_gender.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/crossing-crowdsourcing/blob/main/figures/map_gender.html)  
Map of distribution of gender.

#### Technical characteristics of participants
[![dimensions of browser](figures/scatter_window_width-window_height.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/crossing-crowdsourcing/blob/main/figures/scatter_window_width-window_height.html)  
Dimensions of browser.
