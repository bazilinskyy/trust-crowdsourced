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
[![plot_all_0](figures/AOI_0.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/AOI_0.html)
Plot of AOI Video_0 All Participants
[![plot_all_1](figures/AOI_1.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/AOI_1.html)
Plot of AOI Video_1 All Participants
[![plot_all_2](figures/AOI_2.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/AOI_2.html)
Plot of AOI Video_2 All Participants
[![plot_all_3](figures/AOI_3.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/AOI_3.html)
Plot of AOI Video_3 All Participants
[![plot_all_4](figures/AOI_4.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/AOI_4.html)
Plot of AOI Video_4 All Participants
[![plot_all_5](figures/AOI_5.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/AOI_5.html)
Plot of AOI Video_5 All Participants
[![plot_all_6](figures/AOI_6.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/AOI_6.html)
Plot of AOI Video_6 All Participants
[![plot_all_7](figures/AOI_7.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/AOI_7.html)
Plot of AOI Video_7 All Participants
[![plot_all_8](figures/AOI_8.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/AOI_8.html)
Plot of AOI Video_8 All Participants
[![plot_all_9](figures/AOI_9.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/AOI_9.html)
Plot of AOI Video_9 All Participants
[![plot_all_10](figures/AOI_10.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/AOI_10.html)
Plot of AOI Video_10 All Participants
[![plot_all_11](figures/AOI_11.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/AOI_11.html)
Plot of AOI Video_11 All Participants
[![plot_all_12](figures/AOI_12.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/AOI_12.html)
Plot of AOI Video_12 All Participants
[![plot_all_13](figures/AOI_13.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/AOI_13.html)
Plot of AOI Video_13 All Participants
[![plot_all_14](figures/AOI_14.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/AOI_14.html)
Plot of AOI Video_14 All Participants
[![plot_all_15](figures/AOI_15.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/AOI_15.html)
Plot of AOI Video_15 All Participants
[![plot_all_16](figures/AOI_16.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/AOI_16.html)
Plot of AOI Video_16 All Participants
[![plot_all_17](figures/AOI_17.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/AOI_17.html)
Plot of AOI Video_17 All Participants
[![plot_all_18](figures/AOI_18.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/AOI_18.html)
Plot of AOI Video_18 All Participants
[![plot_all_19](figures/AOI_19.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/AOI_19.html)
Plot of AOI Video_19 All Participants
[![plot_all_20](figures/AOI_20.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/AOI_20.html)
Plot of AOI Video_20 All Participants

### For only lab participants
[![plot_Lab_only_0](figures/Lab_only_AOI_0.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/Lab_only_AOI_0.html)
Plot of AOI Video_0 Lab only Participants
[![plot_Lab_only_1](figures/Lab_only_AOI_1.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/Lab_only_AOI_1.html)
Plot of AOI Video_1 Lab only Participants
[![plot_Lab_only_2](figures/Lab_only_AOI_2.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/Lab_only_AOI_2.html)
Plot of AOI Video_2 Lab only Participants
[![plot_Lab_only_3](figures/Lab_only_AOI_3.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/Lab_only_AOI_3.html)
Plot of AOI Video_3 Lab only Participants
[![plot_Lab_only_4](figures/Lab_only_AOI_4.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/Lab_only_AOI_4.html)
Plot of AOI Video_4 Lab only Participants
[![plot_Lab_only_5](figures/Lab_only_AOI_5.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/Lab_only_AOI_5.html)
Plot of AOI Video_5 Lab only Participants
[![plot_Lab_only_6](figures/Lab_only_AOI_6.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/Lab_only_AOI_6.html)
Plot of AOI Video_6 Lab only Participants
[![plot_Lab_only_7](figures/Lab_only_AOI_7.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/Lab_only_AOI_7.html)
Plot of AOI Video_7 Lab only Participants
[![plot_Lab_only_8](figures/Lab_only_AOI_8.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/Lab_only_AOI_8.html)
Plot of AOI Video_8 Lab only Participants
[![plot_Lab_only_9](figures/Lab_only_AOI_9.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/Lab_only_AOI_9.html)
Plot of AOI Video_9 Lab only Participants
[![plot_Lab_only_10](figures/Lab_only_AOI_10.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/Lab_only_AOI_10.html)
Plot of AOI Video_10 Lab only Participants
[![plot_Lab_only_11](figures/Lab_only_AOI_11.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/Lab_only_AOI_11.html)
Plot of AOI Video_11 Lab only Participants
[![plot_Lab_only_12](figures/Lab_only_AOI_12.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/Lab_only_AOI_12.html)
Plot of AOI Video_12 Lab only Participants
[![plot_Lab_only_13](figures/Lab_only_AOI_13.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/Lab_only_AOI_13.html)
Plot of AOI Video_13 Lab only Participants
[![plot_Lab_only_14](figures/Lab_only_AOI_14.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/Lab_only_AOI_14.html)
Plot of AOI Video_14 Lab only Participants
[![plot_Lab_only_15](figures/Lab_only_AOI_15.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/Lab_only_AOI_15.html)
Plot of AOI Video_15 Lab only Participants
[![plot_Lab_only_16](figures/Lab_only_AOI_16.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/Lab_only_AOI_16.html)
Plot of AOI Video_16 Lab only Participants
[![plot_Lab_only_17](figures/Lab_only_AOI_17.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/Lab_only_AOI_17.html)
Plot of AOI Video_17 Lab only Participants
[![plot_Lab_only_18](figures/Lab_only_AOI_18.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/Lab_only_AOI_18.html)
Plot of AOI Video_18 Lab only Participants
[![plot_Lab_only_19](figures/Lab_only_AOI_19.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/Lab_only_AOI_19.html)
Plot of AOI Video_19 Lab only Participants
[![plot_Lab_only_20](figures/Lab_only_AOI_20.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/Lab_only_AOI_20.html)
Plot of AOI Video_20 Lab only Participants

## Key press (KP)
### All participants
[![plot_all_0](figures/kp_videos_sliders_0_21_42_63.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/smooth-kp-one-euro-filter/figures/kp_videos_sliders_0_21_42_63.html)
Plot of KP Video_0 All Participants
[![plot_all_1](figures/kp_videos_sliders_1_22_43_64.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/smooth-kp-one-euro-filter/figures/kp_videos_sliders_1_22_43_64.html)
Plot of KP Video_1 All Participants
[![plot_all_2](figures/kp_videos_sliders_2_23_44_65.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/smooth-kp-one-euro-filter/figures/kp_videos_sliders_2_23_44_65.html)
Plot of KP Video_2 All Participants
[![plot_all_3](figures/kp_videos_sliders_3_24_45_66.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/smooth-kp-one-euro-filter/figures/kp_videos_sliders_3_24_45_66.html)
Plot of KP Video_3 All Participants
[![plot_all_4](figures/kp_videos_sliders_4_25_46_67.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/smooth-kp-one-euro-filter/figures/kp_videos_sliders_4_25_46_67.html)
Plot of KP Video_4 All Participants
[![plot_all_5](figures/kp_videos_sliders_5_26_47_68.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/smooth-kp-one-euro-filter/figures/kp_videos_sliders_5_26_47_68.html)
Plot of KP Video_5 All Participants
[![plot_all_6](figures/kp_videos_sliders_6_27_48_69.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/smooth-kp-one-euro-filter/figures/kp_videos_sliders_6_27_48_69.html)
Plot of KP Video_6 All Participants
[![plot_all_7](figures/kp_videos_sliders_7_28_49_70.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/smooth-kp-one-euro-filter/figures/kp_videos_sliders_7_28_49_70.html)
Plot of KP Video_7 All Participants
[![plot_all_8](figures/kp_videos_sliders_8_29_50_71.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/smooth-kp-one-euro-filter/figures/kp_videos_sliders_8_29_50_71.html)
Plot of KP Video_8 All Participants
[![plot_all_9](figures/kp_videos_sliders_9_30_51_72.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/smooth-kp-one-euro-filter/figures/kp_videos_sliders_9_30_51_72.html)
Plot of KP Video_9 All Participants
[![plot_all_10](figures/kp_videos_sliders_10_31_52_73.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/smooth-kp-one-euro-filter/figures/kp_videos_sliders_10_31_52_73.html)
Plot of KP Video_10 All Participants
[![plot_all_11](figures/kp_videos_sliders_11_32_53_74.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/smooth-kp-one-euro-filter/figures/kp_videos_sliders_11_32_53_74.html)
Plot of KP Video_11 All Participants
[![plot_all_12](figures/kp_videos_sliders_12_33_54_75.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/smooth-kp-one-euro-filter/figures/kp_videos_sliders_12_33_54_75.html)
Plot of KP Video_12 All Participants
[![plot_all_13](figures/kp_videos_sliders_13_34_55_76.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/smooth-kp-one-euro-filter/figures/kp_videos_sliders_13_34_55_76.html)
Plot of KP Video_13 All Participants
[![plot_all_14](figures/kp_videos_sliders_14_35_56_77.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/smooth-kp-one-euro-filter/figures/kp_videos_sliders_14_35_56_77.html)
Plot of KP Video_14 All Participants
[![plot_all_15](figures/kp_videos_sliders_15_36_57_78.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/smooth-kp-one-euro-filter/figures/kp_videos_sliders_15_36_57_78.html)
Plot of KP Video_15 All Participants
[![plot_all_16](figures/kp_videos_sliders_16_37_58_79.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/smooth-kp-one-euro-filter/figures/kp_videos_sliders_16_37_58_79.html)
Plot of KP Video_16 All Participants
[![plot_all_17](figures/kp_videos_sliders_17_38_59_80.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/smooth-kp-one-euro-filter/figures/kp_videos_sliders_17_38_59_80.html)
Plot of KP Video_17 All Participants
[![plot_all_18](figures/kp_videos_sliders_18_39_60_81.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/smooth-kp-one-euro-filter/figures/kp_videos_sliders_18_39_60_81.html)
Plot of KP Video_18 All Participants
[![plot_all_19](figures/kp_videos_sliders_19_40_61_82.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/smooth-kp-one-euro-filter/figures/kp_videos_sliders_19_40_61_82.html)
Plot of KP Video_19 All Participants
[![plot_all_20](figures/kp_videos_sliders_20_41_62_83.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/smooth-kp-one-euro-filter/figures/kp_videos_sliders_20_41_62_83.html)
Plot of KP Video_20 All Participants

### For only lab participants
[![plot_Lab_only_0](figures/Lab_only_KP_0.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/Lab_only_KP_0.html)
Plot of KP Video_0 Lab only Participants
[![plot_Lab_only_1](figures/Lab_only_KP_1.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/Lab_only_KP_1.html)
Plot of KP Video_1 Lab only Participants
[![plot_Lab_only_2](figures/Lab_only_KP_2.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/Lab_only_KP_2.html)
Plot of KP Video_2 Lab only Participants
[![plot_Lab_only_3](figures/Lab_only_KP_3.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/Lab_only_KP_3.html)
Plot of KP Video_3 Lab only Participants
[![plot_Lab_only_4](figures/Lab_only_KP_4.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/Lab_only_KP_4.html)
Plot of KP Video_4 Lab only Participants
[![plot_Lab_only_5](figures/Lab_only_KP_5.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/Lab_only_KP_5.html)
Plot of KP Video_5 Lab only Participants
[![plot_Lab_only_6](figures/Lab_only_KP_6.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/Lab_only_KP_6.html)
Plot of KP Video_6 Lab only Participants
[![plot_Lab_only_7](figures/Lab_only_KP_7.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/Lab_only_KP_7.html)
Plot of KP Video_7 Lab only Participants
[![plot_Lab_only_8](figures/Lab_only_KP_8.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/Lab_only_KP_8.html)
Plot of KP Video_8 Lab only Participants
[![plot_Lab_only_9](figures/Lab_only_KP_9.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/Lab_only_KP_9.html)
Plot of KP Video_9 Lab only Participants
[![plot_Lab_only_10](figures/Lab_only_KP_10.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/Lab_only_KP_10.html)
Plot of KP Video_10 Lab only Participants
[![plot_Lab_only_11](figures/Lab_only_KP_11.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/Lab_only_KP_11.html)
Plot of KP Video_11 Lab only Participants
[![plot_Lab_only_12](figures/Lab_only_KP_12.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/Lab_only_KP_12.html)
Plot of KP Video_12 Lab only Participants
[![plot_Lab_only_13](figures/Lab_only_KP_13.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/Lab_only_KP_13.html)
Plot of KP Video_13 Lab only Participants
[![plot_Lab_only_14](figures/Lab_only_KP_14.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/Lab_only_KP_14.html)
Plot of KP Video_14 Lab only Participants
[![plot_Lab_only_15](figures/Lab_only_KP_15.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/Lab_only_KP_15.html)
Plot of KP Video_15 Lab only Participants
[![plot_Lab_only_16](figures/Lab_only_KP_16.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/Lab_only_KP_16.html)
Plot of KP Video_16 Lab only Participants
[![plot_Lab_only_17](figures/Lab_only_KP_17.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/Lab_only_KP_17.html)
Plot of KP Video_17 Lab only Participants
[![plot_Lab_only_18](figures/Lab_only_KP_18.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/Lab_only_KP_18.html)
Plot of KP Video_18 Lab only Participants
[![plot_Lab_only_19](figures/Lab_only_KP_19.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/Lab_only_KP_19.html)
Plot of KP Video_19 Lab only Participants
[![plot_Lab_only_20](figures/Lab_only_KP_20.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/trust-crowdsourced/blob/main/figures/Lab_only_KP_20.html)
Plot of KP Video_20 Lab only Participants
