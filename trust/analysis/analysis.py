# by Pavlo Bazilinskyy <pavlo.bazilinskyy@gmail.com>
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import seaborn as sns
import pandas as pd
import plotly as py
import plotly.graph_objs as go
import matplotlib.animation as animation
import subprocess
import io
import pickle
import plotly.express as px
from plotly import subplots
# For OneEuroFilter, see https://github.com/casiez/OneEuroFilter
from OneEuroFilter import OneEuroFilter
import warnings
import unicodedata
import re
from tqdm import tqdm
import ast
from scipy.signal import savgol_filter
from scipy.stats.kde import gaussian_kde
from scipy.stats import ttest_rel, ttest_ind, f_oneway
import cv2
from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols
import trust as tr
import pingouin as pg
from scipy.stats import rankdata
from scipy.stats import wilcoxon, mannwhitneyu
from statsmodels.stats.multitest import multipletests


matplotlib.use('TkAgg')
logger = tr.CustomLogger(__name__)  # use custom logger


class Analysis:
    # set template for plotly output
    template = tr.common.get_configs('plotly_template')
    # store resolution for keypress data
    res = tr.common.get_configs('kp_resolution')
    # number of stimuli
    num_stimuli = tr.common.get_configs('num_stimuli')
    # smoothen signal or not
    smoothen_signal = tr.common.get_configs('smoothen_signal')
    # todo: cleanup code for eye gaze analysis
    fig = None
    g = None
    image = None
    stim_id = None
    points = None
    save_frames = False
    folder_figures = 'figures'  # subdirectory to save figures
    folder_stats = 'statistics'  # subdirectory to save statistical output
    polygons = None

    def __init__(self):
        # set font globally
        plt.rcParams['font.family'] = tr.common.get_configs('font_family')

    def save_all_frames(self, df, mapping, id_video, t):
        """
        Outputs individual frames as png from inputted video mp4.

    Args:
            df (dataframe): dataframe of heroku.
            mapping (TYPE): mapping to extract timestamp.
            id_video (int): stimulus video ID.
            t (list): column in dataframe containing time data.

    Returns:
            None
        """
        logger.info('Creating frames.')
        # path for temp folder to store images with frames
        path = os.path.join(tr.settings.output_dir, 'frames')
        # create temp folder
        if not os.path.exists(path):
            os.makedirs(path)
        # video file in the folder with stimuli
        cap = cv2.VideoCapture(
            os.path.join(
                tr.common.get_configs('path_stimuli'),
                'video_' + str(id_video) + '.mp4'))
        # timestamp
        t = mapping.loc['video_' + str(id_video)][t]
        self.time = int(t)
        self.hm_resolution = tr.common.get_configs('hm_resolution')
        hm_resolution_int = int(tr.common.get_configs('hm_resolution'))
        # check if file is already open
        if not cap.isOpened():
            logger.error('File with frame already open.')
            return
        # go over frames
        for k in tqdm(range(0, self.time, hm_resolution_int)):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.set(cv2.CAP_PROP_POS_FRAMES, round(fps * k / 1000))
            ret, frame = cap.read()
            if ret:
                filename = os.path.join(
                    path,
                    'frame_' + str([round(k / hm_resolution_int)]) + '.jpg')
                cv2.imwrite(filename, frame, [cv2.IMWRITE_JPEG_QUALITY, 20])

    def create_histogram(self, image, points, id_video, density_coef=10, suffix='_histogram.jpg', save_file=False):
        """
        Create histogram for image based on the list of lists of points.
        density_coef: coefficient for division of dimensions for density of points.

    Args:
            image (image): image as the base.
            points (list): data.
            id_video (int): ID of video
            density_coef (int, optional): coefficient for density plot.
            suffix (str, optional): suffix for saved file.
            save_file (bool, optional): whether to save file or not.

    Returns:
            TYPE: Description
        """
        # check if data is present
        if not points:
            logger.error('Not enough data. Histogram was not created for {}.', image)
            return
        # get dimensions of stimulus
        width = tr.common.get_configs('stimulus_width')
        height = tr.common.get_configs('stimulus_height')
        # convert points into np array
        xy = np.array(points)
        # split coordinates list for readability
        x = xy[:, 0]
        y = xy[:, 1]
        # create figure object with given dpi and dimensions
        dpi = 150
        fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
        # build histogram
        plt.hist2d(x=x,
                   y=-y,  # convert to the reference system in image
                   bins=[round(width / density_coef), round(height / density_coef)],
                   cmap=plt.cm.jet)
        plt.colorbar()
        # remove white spaces around figure
        plt.gca().set_axis_off()
        # save video
        if save_file:
            self.save_fig(image, fig, '_video_' + str(id_video) + suffix)

    def create_heatmap(self, image, points, type_heatmap='contourf', add_corners=True, save_file=False):
        """
        Create heatmap for image based on the list of lists of points.

    Args:
            image (image): image as the base.
            points (list): data.
            type_heatmap (str, optional): Type=contourf, pcolormesh, kdeplot.
            add_corners (bool, optional): add points to the corners to have the heatmap overlay the whole image.
            save_file (bool, optional): whether to save file or not.

    Returns:
            fig, g: figure.
        """
        # todo: remove datapoints in corners in heatmaps
        # check if data is present
        if not points:
            logger.error('Not enough data. Heatmap was not created for {}.', image)
            return
        # get dimensions of base image
        width = tr.common.get_configs('stimulus_width')
        height = tr.common.get_configs('stimulus_height')
        # add datapoints to corners for maximised heatmaps
        if add_corners:
            if [0, 0] not in points:
                points.append([0, 0])
            if [width, height] not in points:
                points.append([width - 1, height - 1])
        # convert points into np array
        xy = np.array(points)
        # split coordinates list for readability
        x = xy[:, 0]
        y = xy[:, 1]
        # compute data for the heatmap
        try:
            k = gaussian_kde(np.vstack([x, y]))
            xi, yi = np.mgrid[x.min():x.max():x.size**0.5*1j, y.min():y.max():y.size**0.5*1j]
            zi = k(np.vstack([xi.flatten(), yi.flatten()]))
        except (np.linalg.LinAlgError, np.linalg.LinAlgError, ValueError):
            logger.error('Not enough data. Heatmap was not created for {}.', image)
            return
        # create figure object with given dpi and dimensions
        dpi = 150
        fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
        # alpha=0.5 makes the plot semitransparent
        suffix_file = ''  # suffix to add to saved image
        if type_heatmap == 'contourf':
            try:
                g = plt.contourf(xi, yi, zi.reshape(xi.shape), alpha=0.5)
                plt.margins(0, 0)
                plt.gca().xaxis.set_major_locator(plt.NullLocator())
                plt.gca().yaxis.set_major_locator(plt.NullLocator())
            except TypeError:
                logger.error('Not enough data. Heatmap was not created for {}.', image)
                plt.close(fig)  # clear figure from memory
                return
            suffix_file = '_contourf.jpg'
        elif type_heatmap == 'pcolormesh':
            try:
                g = plt.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='auto', alpha=0.5)
                plt.margins(0, 0)
                plt.gca().xaxis.set_major_locator(plt.NullLocator())
                plt.gca().yaxis.set_major_locator(plt.NullLocator())
            except TypeError:
                logger.error('Not enough data. Heatmap was not created for {}.', image)
                plt.close(fig)  # clear figure from memory
                return
            suffix_file = '_pcolormesh.jpg'
        elif type_heatmap == 'kdeplot':
            try:
                g = sns.kdeplot(x=x, y=y, alpha=0.5, fill=True, cmap="RdBu_r")
            except TypeError:
                logger.error('Not enough data. Heatmap was not created for {}.', image)
                fig.clf()  # clear figure from memory
                return
            suffix_file = '_kdeplot.jpg'
        else:
            logger.error('Wrong type_heatmap {} given.', type_heatmap)
            plt.close(fig)  # clear from memory
            return
        # read original image
        im = plt.imread(image + '\\frame_' + str([1]) + '.jpg')
        plt.imshow(im)
        # remove axis
        plt.gca().set_axis_off()
        # remove white spaces around figure
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        # save image
        if save_file:
            self.save_fig(image, fig, suffix_file)
        # return graph objects
        return fig, g

    def create_animation(self, df, mapping, image, id_video, points, points1, points2, points3, t, save_anim=False,
                         save_frames=False):
        """
        Create animation for image based on the list of lists of points of
        varying duration.

    Args:
            df (dataframe): dataframe with data.
            mapping (dataframe): mapping dataframe.
            image (image): the frames from the stimulus video
            id_video (int): which stimulus video is being used
            points (list): list containing eye-tracking points
            points1 (list): points for stimuli 21 - 41.
            points2 (list): points for stimuli 42 - 62.
            points3 (list): points for stimuli 63 - 83.
            t (int): timestamp.
            save_anim (bool, optional): whether to save animation of not.
            save_frames (bool, optional): whether to save individual frames of not.
        """
        self.image = image
        self.hm_resolution_range = int(50000/tr.common.get_configs('hm_resolution'))
        self.id_video = id_video
        # calc amounts of steps from duration
        # dur = mapping.iloc[id_video]['video_length']
        # Determine the amount of frames for given video
        self.frames = int(round(self.time/self.hm_resolution))
        # Determine time
        self.t = mapping.loc['video_'+str(id_video)][t]
        # Call eye-tracking points
        self.points = points
        self.save_frames = save_frames
        # Create subplot figure with heatmap and kp plot
        self.fig, self.g = plt.subplots(nrows=3,
                                        ncols=1,
                                        figsize=(20, 20),
                                        gridspec_kw=dict(height_ratios=[1, 1, 3], hspace=0.2))
        self.fig.suptitle('Keypresses and eye-tracking heatmap video_' + str(self.id_video), fontsize=30)
        # Deterin time and data for kp plot
        self.times = np.array(range(self.res, mapping['video_length'].max() + self.res, self.res)) / 1000
        self.kp_data = mapping.loc['video_' + str(id_video)]['kp']
        self.event = mapping.loc['video_' + str(id_video)]['events']
        self.event = re.findall(r'\w+', self.event)
        aoi = pd.read_csv(tr.common.get_configs('aoi'))
        aoi.set_index('video_id', inplace=True)
        self.number_in = []
        # for combined animation
        self.number_in1 = []
        self.number_in2 = []
        self.number_in3 = []
        # for comparison between stimuli stim 21 - 41
        self.kp_data1 = mapping.loc['video_' + str(id_video+21)]['kp']
        self.points1 = points1
        # stim 42 - 62
        self.kp_data2 = mapping.loc['video_' + str(id_video+42)]['kp']
        self.points2 = points2
        # stim 63 - 83
        self.kp_data3 = mapping.loc['video_' + str(id_video+63)]['kp']
        self.points3 = points3
        # extracting AOI coordinate data
        self.aoit = []
        self.aoi_x = aoi.loc['video_' + str(id_video)]['x']
        self.aoi_x = self.aoi_x.split(", ")
        self.aoi_y = aoi.loc['video_' + str(id_video)]['y']
        self.aoi_y = self.aoi_y.split(", ")
        self.aoi_t = aoi.loc['video_' + str(id_video)]['t']
        self.aoi_t = self.aoi_t.split(", ")
        # event description for in the animation plots
        self.event_discription = re.split(',', mapping.loc['video_' + str(id_video)]['events_description'])
        # animate frames subplots into one animation using animate function
        anim = animation.FuncAnimation(self.fig,
                                       self.animate,
                                       frames=self.frames,
                                       interval=self.hm_resolution,
                                       repeat=False)
        # save image
        if save_anim:
            self.save_anim(image, anim, '_video_' + str(id_video) + '_animation.mp4')

    def create_animation_all_stimuli(self, num_stimuli):
        """
        Create long video with all animations.

    Args:
            num_stimuli (int): number of stimuli.
        """
        logger.info('Creating long video with all animations for {} stimuli.', num_stimuli)
        # create path
        path = os.path.join(tr.settings.output_dir, self.folder_figures)
        if not os.path.exists(path):
            os.makedirs(path)
        # file with list of animations
        list_anim = path + 'animations.txt'
        file = open(list_anim, 'w+')
        # loop of stimuli
        for id_video in range(1, num_stimuli - 1):
            # add animation to the list
            anim_path = path + '_video_' + str(id_video) + '_animation.mp4'
            # check if need to add a linebreak
            if id_video == num_stimuli:
                file.write('file ' + anim_path)  # no need for linebreak
            else:
                file.write('file ' + anim_path + '\n')
        # close file with animations
        file.close()
        # stitch videos together
        os.chdir(path)
        subprocess.call(['ffmpeg',
                         '-y',
                         '-loglevel', 'quiet',
                         '-f', 'concat',
                         '-safe', '0',
                         '-i', list_anim,
                         '-c', 'copy',
                         'all_animations.mp4'])
        # delete file with animations
        os.remove(list_anim)

    def animate(self, i):
        """
        Helper function to create animation.

    Args:
            i (int): ID.

    Returns:
            figure: figure object.
        """
        self.g[0].clear()
        self.g[1].clear()
        self.g[2].clear()
        durations = range(0, self.hm_resolution_range)
        # Subplot 1 KP data
        it = int(round(len(self.kp_data)*i/(self.frames)))
        self.g[0].plot(np.array(self.times[:it]),
                       np.array(self.kp_data[:it]),
                       lw=1,
                       label='Video_' + str(self.id_video),
                       color='r')
        # If animations are combined scenarios
        if tr.common.get_configs('combined_animation') == 1:
            self.g[0].plot(np.array(self.times[:it]),
                           np.array(self.kp_data1[:it]),
                           lw=1,
                           label='Video_' + str(self.id_video+21),
                           color='b')
            self.g[0].plot(np.array(self.times[:it]),
                           np.array(self.kp_data2[:it]),
                           lw=1,
                           label='Video_' + str(self.id_video+42),
                           color='g')
            self.g[0].plot(np.array(self.times[:it]),
                           np.array(self.kp_data3[:it]),
                           lw=1,
                           label='Video_' + str(self.id_video+63),
                           color='m')
        # Adding legend and formating to figure
        self.g[0].legend()
        self.g[0].set_xlabel("Time (s)", fontsize=15)
        self.g[0].set_ylabel("Percentage of Keypresses", fontsize=15)
        self.g[0].set_xlim(0, 50)
        self.g[0].set_title('Number of keypresses', fontsize=25)
        # Extract time stamps for events from appen data to dislay in plot
        length = int(len(self.event))
        # Plot event lines in kp and aoi plot
        for ev in range(len(self.event)):
            self.g[0].axvline(x=int(self.event[ev])/1000,
                              label="" + str(self.event_discription[ev]),
                              c=plt.cm.RdYlBu(int(ev)/length),
                              lw=2)
            self.g[0].tick_params(axis='x')
            self.g[0].legend(fontsize=15)
            self.g[1].axvline(x=int(self.event[ev])/1000,
                              label="" + str(self.event_discription[ev]),
                              c=plt.cm.RdYlBu(int(ev)/length),
                              lw=2)
            self.g[1].tick_params(axis='x')
            self.g[1].legend(fontsize=15)
        # Subplot 2 AOI
        self.g[1].set_title('Number of eye gazes in area of interest', fontsize=25)
        self.g[1].set_xlabel('Time (s)', fontsize=15)
        self.g[1].set_ylabel('Number of gazes in Area of Interest', fontsize=15)
        if tr.common.get_configs('only_lab') == 1:
            self.g[1].set_ylim(0, 35)
            self.g[0].set_ylim(0, 80)
        else:
            self.g[1].set_ylim(0, 600)
            self.g[0].set_ylim(0, 50)
        self.g[1].set_xlim(0, 50)
        # AOI data
        aoi_x = float(self.aoi_x[i])
        aoi_y = float(self.aoi_y[i])
        aoi_t = float(self.aoi_t[i])
        self.aoit.append(int(aoi_t)/1000)
        # Defining boundaries of AOI
        min_x = int(aoi_x) - 100
        max_x = int(aoi_x) + 100
        min_y = int(aoi_y) - 100
        max_y = int(aoi_y) + 100
        # stim 0 - 20 or all stim when not combined
        x = [item[0] for item in self.points[i]]
        y = [item[1] for item in self.points[i]]
        if tr.common.get_configs('combined_animation') == 1:
            # stim 21 - 41
            x1 = [item[0] for item in self.points1[i]]
            y1 = [item[1] for item in self.points1[i]]
            # stim 42 - 62
            x2 = [item[0] for item in self.points2[i]]
            y2 = [item[1] for item in self.points2[i]]
            # stim 63 - 83
            x3 = [item[0] for item in self.points3[i]]
            y3 = [item[1] for item in self.points3[i]]
            # Filtering data for inside or outside coordinates
            num1 = 0
            num2 = 0
            num3 = 0
            for v in range(len(x1)):
                if max_x > x1[v] > min_x:
                    if max_y > y1[v] > min_y:
                        num1 = num1 + 1
                    else:
                        continue
                else:
                    continue
            for v in range(len(x2)):
                if max_x > x2[v] > min_x:
                    if max_y > y2[v] > min_y:
                        num2 = num2 + 1
                    else:
                        continue
                else:
                    continue
            for v in range(len(x3)):
                if max_x > x3[v] > min_x:
                    if max_y > y3[v] > min_y:
                        num3 = num3 + 1
                    else:
                        continue
                else:
                    continue
            if i < 10:
                self.number_in1.append(int(num1))
                number_in_plot1 = self.number_in1
                self.number_in2.append(int(num2))
                number_in_plot2 = self.number_in2
                self.number_in3.append(int(num3))
                number_in_plot3 = self.number_in3

            else:
                self.number_in1 = np.append(self.number_in1, int(num1))
                number_in_plot1 = savgol_filter(self.number_in1, 10, 2)
                self.number_in2 = np.append(self.number_in2, int(num2))
                number_in_plot2 = savgol_filter(self.number_in2, 10, 2)
                self.number_in3 = np.append(self.number_in3, int(num3))
                number_in_plot3 = savgol_filter(self.number_in3, 10, 2)
            # plot AOI gazes
            self.g[1].plot(self.aoit,
                           number_in_plot1,
                           label='Video_' + str(self.id_video+21),
                           color='b')
            self.g[1].plot(self.aoit,
                           number_in_plot2,
                           label='Video_' + str(self.id_video+42),
                           color='g')
            self.g[1].plot(self.aoit,
                           number_in_plot3,
                           label='Video_' + str(self.id_video+63),
                           color='m')
        # Filtering data for if they are inside or outside coordinates
        num = 0
        for v in range(len(x)):
            if max_x > x[v] > min_x:
                if max_y > y[v] > min_y:
                    num = num + 1
                else:
                    continue
            else:
                continue
        if i < 10:
            self.number_in.append(int(num))
            number_in_plot = self.number_in
        else:
            self.number_in = np.append(self.number_in, int(num))
            number_in_plot = savgol_filter(self.number_in, 10, 2)
        self.g[1].plot(self.aoit,
                       number_in_plot,
                       label='Video_' + str(self.id_video),
                       color='r')
        # add legned for figure
        self.g[1].legend(fontsize=15)
        # Subplot 3 Heatmap
        self.g[2] = sns.kdeplot(x=[item[0] for item in self.points[i]],
                                y=[item[1] for item in self.points[i]],
                                alpha=0.5,
                                fill=True,
                                cmap='RdBu_r')
        self.g[2].invert_yaxis()
        self.g[2].plot([min_x, max_x, max_x, min_x, min_x], [min_y, min_y, max_y, max_y, min_y], color="red")
        if i == self.frames-1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=np.array(self.times[:it]),
                                     y=np.array(self.kp_data[:it]),
                                     mode='lines',
                                     name='video_' + str(self.id_video)))
            fig.add_trace(go.Scatter(x=np.array(self.times[:it]),
                                     y=np.array(self.kp_data1[:it]),
                                     mode='lines',
                                     name='video_' + str(self.id_video+21)))
            fig.add_trace(go.Scatter(x=np.array(self.times[:it]),
                                     y=np.array(self.kp_data2[:it]),
                                     mode='lines',
                                     name='video_' + str(self.id_video+42)))
            fig.add_trace(go.Scatter(x=np.array(self.times[:it]),
                                     y=np.array(self.kp_data3[:it]),
                                     mode='lines',
                                     name='video_' + str(self.id_video+63)))
            fig.update_layout(template=self.template,
                              xaxis_title='time(ms)',
                              yaxis_title="Number of KP")
            file_name = 'lab_only_kp_' + str(self.id_video)
            self.save_plotly(fig=fig, name=file_name)
        # Scatter plot data
        # all pp
        # self.g = sns.scatterplot(x=[item[0] for item in self.points[i]],
        #                          y=[item[1] for item in self.points[i]],
        #                          alpha=0.5,
        #                          hue=[item[0] for item in self.points[i]],
        #                          legend='auto')
        # read original image
        path = self.image
        im = plt.imread(os.path.join(path, "frame_" + str([i]) + ".jpg"))
        plt.imshow(im)

        # remove axis
        plt.gca().set_axis_off()
        # remove white spaces around figure
        plt.subplots_adjust(top=1,
                            bottom=0,
                            right=1,
                            left=0,
                            hspace=0,
                            wspace=0)
        # textbox with duration
        # props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        # plt.text(0.75,
        #          0.98,
        #          'id_video=' + str(self.id_video) +
        #          ' time (ms)=' + str(round(durations[i]*int(self.t)/self.hm_resolution)),
        #          transform=plt.gca().transAxes,
        #          fontsize=12,
        #          verticalalignment='top',
        #          bbox=props)
        # save each frame as file
        if self.save_frames:
            # build filename
            name = '_kdeplot_' + str(durations[i]) + '.jpg'
            # copy figure in buffer to prevent destruction of object
            buf = io.BytesIO()
            pickle.dump(self.fig, buf)
            buf.seek(0)
            temp_fig = pickle.load(buf)
            # save figure
            self.save_fig(self.image, temp_fig, os.path.join(tr.settings.output_dir, self.folder_figures), name)
        return self.g
        # save each frame as file
        if self.save_frames:
            # build filename
            name = '_kdeplot_' + str(durations[i]) + '.jpg'
            # copy figure in buffer to prevent destruction of object
            buf = io.BytesIO()
            pickle.dump(self.fig, buf)
            buf.seek(0)
            temp_fig = pickle.load(buf)
            # save figure
            self.save_fig(self.image, temp_fig, name)
        return self.g

    def corr_matrix(self, df, columns_drop, name_file='corr_matrix.jpg', save_file=False, save_final=False):
        """
        Output correlation matrix.

        Args:
            df (dataframe): mapping dataframe.
            columns_drop (list): columns dataframes in to ignore.
            name_file (str, optional): name of file to save.
            save_file (bool, optional): flag for saving an html file with plot.
            save_final (bool, optional): flag for saving an a final figure to /figures.
        """
        logger.info('Creating correlation matrix.')
        # drop columns
        df = df.drop(columns=columns_drop)
        # create correlation matrix
        corr = df.corr()
        # create mask
        mask = np.zeros_like(corr)
        mask[np.triu_indices_from(mask)] = True
        # set larger font
        # vs_font = 12  # very small
        # s_font = 14   # small
        # m_font = 18   # medium
        l_font = 24   # large
        plt.rc('font', size=l_font)         # controls default text sizes
        # plt.rc('axes', titlesize=l_font)    # fontsize of the axes title
        # plt.rc('axes', labelsize=l_font)    # fontsize of the axes labels
        # plt.rc('xtick', labelsize=l_font)  # fontsize of the tick labels
        # plt.rc('ytick', labelsize=l_font)  # fontsize of the tick labels
        # plt.rc('legend', fontsize=l_font)   # fontsize of the legend
        # plt.rc('figure', titlesize=l_font)  # fontsize of the figure title
        # plt.rc('axes', titlesize=l_font)    # fontsize of the subplot title
        # create figure
        fig = plt.figure(figsize=(34, 20))
        g = sns.heatmap(corr, annot=True, mask=mask, cmap='coolwarm', fmt=".2f")
        # rotate ticks
        for item in g.get_xticklabels():
            item.set_rotation(55)
        # save file to local output folder
        if save_file:
            self.save_fig(fig=fig,
                          name=name_file,
                          pad_inches=0.05,
                          save_final=save_final)  # also save as "final" figure
        # open it in localhost instead
        else:
            fig.show()
        # revert font
        self.reset_font()

    def scatter_matrix(self, df, columns_drop, color=None, symbol=None, diagonal_visible=False, xaxis_title=None,
                       yaxis_title=None, name_file='scatter_matrix', save_file=False, save_final=False,
                       fig_save_width=1320, fig_save_height=680, font_family=None, font_size=None):
        """
        Output scatter matrix.

        Args:
            df (dataframe): mapping dataframe.
            columns_drop (list): columns dataframes in to ignore.
            color (str, optional): dataframe column to assign colour of points.
            symbol (str, optional): dataframe column to assign symbol of points.
            diagonal_visible (bool, optional): show/hide diagonal with correlation==1.0.
            xaxis_title (str, optional): title for x axis.
            yaxis_title (str, optional): title for y axis.
            name_file (str, optional): name of file to save.
            save_file (bool, optional): flag for saving an html file with plot.
            save_final (bool, optional): flag for saving an a final figure to /figures.
            fig_save_width (int, optional): width of figures to be saved.
            fig_save_height (int, optional): height of figures to be saved.
            font_family (str, optional): font family to be used across the figure. None = use config value.
            font_size (int, optional): font size to be used across the figure. None = use config value.
        """
        logger.info('Creating scatter matrix.')
        # drop columns
        df = df.drop(columns=columns_drop)
        # create dimensions list after dropping columns
        dimensions = df.keys()
        # plot matrix
        fig = px.scatter_matrix(df, dimensions=dimensions, color=color, symbol=symbol)
        # update layout
        fig.update_layout(template=self.template,
                          width=5000,
                          height=5000,
                          xaxis_title=xaxis_title,
                          yaxis_title=yaxis_title)
        # hide diagonal
        if not diagonal_visible:
            fig.update_traces(diagonal_visible=False)
        # update font family
        if font_family:
            # use given value
            fig.update_layout(font=dict(family=font_family))
        else:
            # use value from config file
            fig.update_layout(font=dict(family=tr.common.get_configs('font_family')))
        # update font size
        if font_size:
            # use given value
            fig.update_layout(font=dict(size=font_size))
        else:
            # use value from config file
            fig.update_layout(font=dict(size=tr.common.get_configs('font_size')))
        # save file to local output folder
        if save_file:
            self.save_plotly(fig=fig,
                             name=name_file,
                             remove_margins=True,
                             width=fig_save_width,
                             height=fig_save_height,
                             save_final=save_final)  # also save as "final" figure
        # open it in localhost instead
        else:
            fig.show()

    def bar(self, df, y: list, y_legend=None, x=None, stacked=False, pretty_text=False, orientation='v',
            xaxis_title=None, yaxis_title=None, show_all_xticks=False, show_all_yticks=False, show_text_labels=False,
            font_family=None, font_size=None, name_file=None, save_file=False, save_final=False, fig_save_width=1320,
            fig_save_height=680):
        """
        Barplot for questionnaire data. Passing a list with one variable will output a simple barplot; passing a list
        of variables will output a grouped barplot.

        Args:
            df (dataframe): dataframe with stimuli data.
            y (list): column names of dataframe to plot.
            y_legend (list, optional): names for variables to be shown in the legend.
            x (list): values in index of dataframe to plot for. If no value is given, the index of df is used.
            stacked (bool, optional): show as stacked chart.
            pretty_text (bool, optional): prettify ticks by replacing _ with spaces and capitalising each value.
            orientation (str, optional): orientation of bars. v=vertical, h=horizontal.
            xaxis_title (str, optional): title for x axis.
            yaxis_title (str, optional): title for y axis.
            show_all_xticks (bool, optional): show all ticks on x axis.
            show_all_yticks (bool, optional): show all ticks on y axis.
            show_text_labels (bool, optional): output automatically positioned text labels.
            name_file (str, optional): name of file to save.
            save_file (bool, optional): flag for saving an html file with plot.
            save_final (bool, optional): flag for saving an a final figure to /figures.
            fig_save_width (int, optional): width of figures to be saved.
            fig_save_height (int, optional): height of figures to be saved.
            font_family (str, optional): font family to be used across the figure. None = use config value.
            font_size (int, optional): font size to be used across the figure. None = use config value.
        """
        logger.info('Creating bar chart for x={} and y={}.', x, y)
        # prettify text
        if pretty_text:
            for variable in y:
                # check if column contains strings
                if isinstance(df.iloc[0][variable], str):
                    # replace underscores with spaces
                    df[variable] = df[variable].str.replace('_', ' ')
                    # capitalise
                    df[variable] = df[variable].str.capitalize()
        # use index of df if no is given
        if not x:
            x = df.index
        # create figure
        fig = go.Figure()
        # go over variables to plot
        for variable in range(len(y)):
            # showing text labels
            if show_text_labels:
                text = df[y[variable]]
            else:
                text = None
            # custom labels for legend
            if y_legend:
                name = y_legend[variable]
            else:
                name = y[variable]
            # plot variable
            fig.add_trace(go.Bar(x=x,
                                 y=df[y[variable]],
                                 name=name,
                                 orientation=orientation,
                                 text=text,
                                 textposition='auto'))
        # add tabs if multiple variables are plotted
        if len(y) > 1:
            fig.update_layout(barmode='group')
            buttons = list([dict(label='All',
                                 method='update',
                                 args=[{'visible': [True] * df[y].shape[0]},
                                       {'title': 'All', 'showlegend': True}])])
            # counter for traversing through stimuli
            counter_rows = 0
            for variable in y:
                visibility = [[counter_rows == j] for j in range(len(y))]
                visibility = [item for sublist in visibility for item in sublist]  # type: ignore
                button = dict(label=variable,
                              method='update',
                              args=[{'visible': visibility},
                                    {'title': variable}])
                buttons.append(button)
                counter_rows = counter_rows + 1
            updatemenus = [dict(x=-0.15, buttons=buttons, showactive=True)]
            fig['layout']['updatemenus'] = updatemenus
            fig['layout']['title'] = 'All'
        # update layout
        fig.update_layout(template=self.template, xaxis_title=xaxis_title, yaxis_title=yaxis_title)
        # format text labels
        if show_text_labels:
            fig.update_traces(texttemplate='%{text:.2f}')
        # show all ticks on x axis
        if show_all_xticks:
            fig.update_layout(xaxis=dict(dtick=1))
        # show all ticks on x axis
        if show_all_yticks:
            fig.update_layout(yaxis=dict(dtick=1))
        # stacked bar chart
        if stacked:
            fig.update_layout(barmode='stack')
        # update font family
        if font_family:
            # use given value
            fig.update_layout(font=dict(family=font_family))
        else:
            # use value from config file
            fig.update_layout(font=dict(family=tr.common.get_configs('font_family')))
        # update font size
        if font_size:
            # use given value
            fig.update_layout(font=dict(size=font_size))
        else:
            # use value from config file
            fig.update_layout(font=dict(size=tr.common.get_configs('font_size')))
        # save file to local output folder
        if save_file:
            # build filename
            if not name_file:
                name_file = 'bar_' + '-'.join(str(val) for val in y) + '_' + '-'.join(str(val) for val in x)
            self.save_plotly(fig=fig,
                             name=name_file,
                             remove_margins=True,
                             width=fig_save_width,
                             height=fig_save_height,
                             save_final=save_final)  # also save as "final" figure
        # open it in localhost instead
        else:
            fig.show()

    def scatter(self, df, x, y, color=None, symbol=None, size=None, text=None, trendline=None, hover_data=None,
                marker_size=None, pretty_text=False, marginal_x='violin', marginal_y='violin', xaxis_title=None,
                yaxis_title=None, xaxis_range=None, yaxis_range=None, name_file=None, save_file=False,
                save_final=False, fig_save_width=1320, fig_save_height=680, font_family=None, font_size=None):
        """
        Output scatter plot of variables x and y with optional assignment of colour and size.

        Args:
            df (dataframe): dataframe with data from heroku.
            x (str): dataframe column to plot on x axis.
            y (str): dataframe column to plot on y axis.
            color (str, optional): dataframe column to assign colour of points.
            symbol (str, optional): dataframe column to assign symbol of points.
            size (str, optional): dataframe column to assign doze of points.
            text (str, optional): dataframe column to assign text labels.
            trendline (str, optional): trendline. Can be 'ols', 'lowess'
            hover_data (list, optional): dataframe columns to show on hover.
            marker_size (int, optional): size of marker. Should not be used together with size argument.
            pretty_text (bool, optional): prettify ticks by replacing _ with spaces and capitalising each value.
            marginal_x (str, optional): type of marginal on x axis. Can be 'histogram', 'rug', 'box', or 'violin'.
            marginal_y (str, optional): type of marginal on y axis. Can be 'histogram', 'rug', 'box', or 'violin'.
            xaxis_title (str, optional): title for x axis.
            yaxis_title (str, optional): title for y axis.
            xaxis_range (list, optional): range of x axis in format [min, max].
            yaxis_range (list, optional): range of y axis in format [min, max].
            name_file (str, optional): name of file to save.
            save_file (bool, optional): flag for saving an html file with plot.
            save_final (bool, optional): flag for saving an a final figure to /figures.
            fig_save_width (int, optional): width of figures to be saved.
            fig_save_height (int, optional): height of figures to be saved.
            font_family (str, optional): font family to be used across the figure. None = use config value.
            font_size (int, optional): font size to be used across the figure. None = use config value.
        """
        logger.info('Creating scatter plot for x={} and y={}.', x, y)
        # using size and marker_size is not supported
        if marker_size and size:
            logger.error('Arguments marker_size and size cannot be used together.')
            return -1
        # using marker_size with histogram marginal(s) is not supported
        if (marker_size and (marginal_x == 'histogram' or marginal_y == 'histogram')):
            logger.error('Argument marker_size cannot be used together with histogram marginal(s).')
            return -1
        # prettify text
        if pretty_text:
            if isinstance(df.iloc[0][x], str):  # check if string
                # replace underscores with spaces
                df[x] = df[x].str.replace('_', ' ')
                # capitalise
                df[x] = df[x].str.capitalize()
            if isinstance(df.iloc[0][y], str):  # check if string
                # replace underscores with spaces
                df[y] = df[y].str.replace('_', ' ')
                # capitalise
                df[y] = df[y].str.capitalize()
            if color and isinstance(df.iloc[0][color], str):  # check if string
                # replace underscores with spaces
                df[color] = df[color].str.replace('_', ' ')
                # capitalise
                df[color] = df[color].str.capitalize()
            if size and isinstance(df.iloc[0][size], str):  # check if string
                # replace underscores with spaces
                df[size] = df[size].str.replace('_', ' ')
                # capitalise
                df[size] = df[size].str.capitalize()
            try:
                # check if string
                if text and isinstance(df.iloc[0][text], str):
                    # replace underscores with spaces
                    df[text] = df[text].str.replace('_', ' ')
                    # capitalise
                    df[text] = df[text].str.capitalize()
            except ValueError as e:
                logger.debug('Tried to prettify {} with exception {}.', text, e)
        # scatter plot with histograms
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            fig = px.scatter(df,
                             x=x,
                             y=y,
                             color=color,
                             symbol=symbol,
                             size=size,
                             text=text,
                             trendline=trendline,
                             hover_data=hover_data,
                             marginal_x=marginal_x,
                             marginal_y=marginal_y)
        # update layout
        fig.update_layout(template=self.template,
                          xaxis_title=xaxis_title,
                          yaxis_title=yaxis_title,
                          xaxis_range=xaxis_range,
                          yaxis_range=yaxis_range)
        # change marker size
        if marker_size:
            fig.update_traces(marker=dict(size=marker_size))
        # update font family
        if font_family:
            # use given value
            fig.update_layout(font=dict(family=font_family))
        else:
            # use value from config file
            fig.update_layout(font=dict(family=tr.common.get_configs('font_family')))
        # update font size
        if font_size:
            # use given value
            fig.update_layout(font=dict(size=font_size))
        else:
            # use value from config file
            fig.update_layout(font=dict(size=tr.common.get_configs('font_size')))
        # save file to local output folder
        if save_file:
            # build filename
            if not name_file:
                name_file = 'scatter_' + x + '-' + y
            self.save_plotly(fig=fig,
                             name=name_file,
                             remove_margins=True,
                             width=fig_save_width,
                             height=fig_save_height,
                             save_final=save_final)  # also save as "final" figure
        # open it in localhost instead
        else:
            fig.show()

    def scatter_mult(self, df, x, y, color=None, symbol=None, text=None, trendline=None, hover_data=None,
                     marker_size=None, pretty_text=False, marginal_x='violin', marginal_y='violin', xaxis_title=None,
                     yaxis_title=None, xaxis_range=None, yaxis_range=None, name_file=None, save_file=False,
                     save_final=False, fig_save_width=1320, fig_save_height=680, font_family=None, font_size=None):
        """
        Output scatter plot of multiple variables x and y with optional assignment of colour and size.

        Args:
            df (dataframe): dataframe with data from heroku.
            x (str): dataframe columns to plot on x axis.
            y (str): dataframe column to plot on y axis.
            symbol (str, optional): dataframe column to assign symbol of points.
            text (str, optional): dataframe column to assign text labels.
            trendline (str, optional): trendline. Can be 'ols', 'lowess'
            hover_data (list, optional): dataframe columns to show on hover.
            marker_size (int, optional): size of marker. Should not be used together with size argument.
            pretty_text (bool, optional): prettify ticks by replacing _ with spaces and capitalising each value.
            marginal_x (str, optional): type of marginal on x axis. Can be 'histogram', 'rug', 'box', or 'violin'.
            marginal_y (str, optional): type of marginal on y axis. Can be 'histogram', 'rug', 'box', or 'violin'.
            xaxis_title (str, optional): title for x axis.
            yaxis_title (str, optional): title for y axis.
            xaxis_range (list, optional): range of x axis in format [min, max].
            yaxis_range (list, optional): range of y axis in format [min, max].
            name_file (str, optional): name of file to save.
            save_file (bool, optional): flag for saving an html file with plot.
            save_final (bool, optional): flag for saving an a final figure to /figures.
            fig_save_width (int, optional): width of figures to be saved.
            fig_save_height (int, optional): height of figures to be saved.
            font_family (str, optional): font family to be used across the figure. None = use config value.
            font_size (int, optional): font size to be used across the figure. None = use config value.
        """
        # todo: extend with multiple columns for y
        logger.info('Creating scatter plot for x={} and y={}.', x, y)
        # using marker_size with histogram marginal(s) is not supported
        if (marker_size and
                (marginal_x == 'histogram' or marginal_y == 'histogram')):
            logger.error('Argument marker_size cannot be used together with histogram marginal(s).')
            return -1
        # prettify text
        if pretty_text:
            for x_col in x:
                if isinstance(df.iloc[0][x_col], str):  # check if string
                    # replace underscores with spaces
                    df[x_col] = df[x_col].str.replace('_', ' ')
                    # capitalise
                    df[x_col] = df[x_col].str.capitalize()
                else:
                    logger.error('No string.')
            if isinstance(df.iloc[0][y], str):  # check if string
                # replace underscores with spaces
                df[y] = df[y].str.replace('_', ' ')
                # capitalise
                df[y] = df[y].str.capitalize()
            else:
                logger.error('No string.')
            try:
                # check if string
                if text and isinstance(df.iloc[0][text], str):
                    # replace underscores with spaces
                    df[text] = df[text].str.replace('_', ' ')
                    # capitalise
                    df[text] = df[text].str.capitalize()
            except ValueError as e:
                logger.debug('Tried to prettify {} with exception {}', text, e)
        # create new dataframe with the necessary data
        color = []
        val_y = []
        val_x = []
        for x_col in x:
            for index, row in df.iterrows():
                color.append(x_col)
                val_x.append(row[x_col])
                val_y.append(row[y])
        data = {'val_y': val_y,
                'color': color,
                'val_x': val_x}
        df = pd.DataFrame(data)
        # scatter plot with histograms
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            fig = px.scatter(df,
                             x='val_x',
                             y='val_y',
                             color='color',
                             symbol=symbol,
                             text=text,
                             trendline=trendline,
                             # hover_data=hover_data,
                             marginal_x=marginal_x,
                             marginal_y=marginal_y)
        # update layout
        fig.update_layout(template=self.template,
                          xaxis_title=xaxis_title,
                          yaxis_title=yaxis_title,
                          xaxis_range=xaxis_range,
                          yaxis_range=yaxis_range,
                          legend_title_text=' ',
                          font=dict(size=20),
                          legend=dict(orientation='h',
                                      yanchor='bottom',
                                      y=1.02,
                                      xanchor='right',
                                      x=0.78))
        # change marker size
        if marker_size:
            fig.update_traces(marker=dict(size=marker_size))
        # update font family
        if font_family:
            # use given value
            fig.update_layout(font=dict(family=font_family))
        else:
            # use value from config file
            fig.update_layout(font=dict(family=tr.common.get_configs('font_family')))
        # update font size
        if font_size:
            # use given value
            fig.update_layout(font=dict(size=font_size))
        else:
            # use value from config file
            fig.update_layout(font=dict(size=tr.common.get_configs('font_size')))
        # save file to local output folder
        if save_file:
            # build filename
            if not name_file:
                name_file = 'scatter_' + ','.join(x) + '-' + y
            self.save_plotly(fig=fig,
                             name=name_file,
                             remove_margins=True,
                             width=fig_save_width,
                             height=fig_save_height,
                             save_final=save_final)  # also save as "final" figure
        # open it in localhost instead
        else:
            fig.show()

    def scatter_et(self, df, x, y, t, pp, id_video, pretty_text=False, marginal_x='violin', marginal_y='violin',
                   xaxis_title=None, xaxis_range=True, yaxis_title=None, yaxis_range=True, name_file=None,
                   save_file=False, save_final=False, fig_save_width=1320, fig_save_height=680, font_family=None,
                   font_size=None):
        """
        Output scatter plot of x and y.

        Args:
            df (dataframe): dataframe with data from Heroku.
            x (list): dataframe column to plot on x axis.
            y (list): dataframe column to plot on y axis.
            t (list): dataframe column to determine timespan
            pp (int): participant ID.
            id_video (int): stimulus video ID.
            pretty_text (bool, optional): prettify ticks by replacing _ with spaces and capitalising each value.
            marginal_x (str, optional): type of marginal on x axis. Can be 'histogram', 'rug', 'box', or 'violin'.
            marginal_y (str, optional): type of marginal on y axis. Can be 'histogram', 'rug', 'box', or 'violin'.
            xaxis_range (list, optional): range of the x-axis plot.
            yaxis_range (list, optional): range of the y-axis plot.
            xaxis_title (str, optional): title for x axis.
            yaxis_title (str, optional): title for y axis.
            name_file (str, optional): name of file to save.
            save_file (bool, optional): flag for saving an html file with plot.
            save_final (bool, optional): flag for saving an a final figure to /figures.
            fig_save_width (int, optional): width of figures to be saved.
            fig_save_height (int, optional): height of figures to be saved.
            font_family (str, optional): font family to be used across the figure. None = use config value.
            font_size (int, optional): font size to be used across the figure. None = use config value.
        """
        logger.info('Creating scatter_map for x={} and y={}.', x, y)
        # extracting x and y values for given ID participant
        width = tr.common.get_configs('stimulus_width')
        height = tr.common.get_configs('stimulus_height')
        x = df.loc[pp][x]
        y = df.loc[pp][y]
        # normalise screen size
        xmin, xmax = min(x), max(x)
        for i, val in enumerate(x):
            x[i] = ((val-xmin) / (xmax-xmin))*width

        ymin, ymax = min(y), max(y)
        for i, val in enumerate(y):
            y[i] = ((val-ymin) / (ymax-ymin))*height
        t = df.loc[pp][t]
        pp = str(pp)
        # Plot animation scatter
        fig = px.scatter(df,
                         x=x,
                         y=y,
                         width=width,
                         height=height,
                         animation_frame=t,
                         marginal_x='violin',
                         marginal_y='violin',
                         title='scatter_' + ' ' + id_video + ' ' + 'participant' + ' ' + pp)
        # update layout
        fig.update_layout(template=self.template,
                          xaxis_title=xaxis_title,
                          yaxis_title=yaxis_title,
                          xaxis_range=[0, width],
                          yaxis_range=[0, height])
        # update font family
        if font_family:
            # use given value
            fig.update_layout(font=dict(family=font_family))
        else:
            # use value from config file
            fig.update_layout(font=dict(family=tr.common.get_configs('font_family')))
        # update font size
        if font_size:
            # use given value
            fig.update_layout(font=dict(size=font_size))
        else:
            # use value from config file
            fig.update_layout(font=dict(size=tr.common.get_configs('font_size')))
        # save file to local output folder
        if save_file:
            # build filename
            if not name_file:
                name_file = 'scatter_map_' + id_video+'_participant_' + pp
            self.save_plotly(fig=fig,
                             name=name_file,
                             remove_margins=True,
                             width=fig_save_width,
                             height=fig_save_height,
                             save_final=save_final)  # also save as "final" figure
        # open it in localhost instead
        else:
            fig.show()

    def heatmap(self, df, x, y, t, id_video, pp, pretty_text=False, marginal_x='violin', marginal_y='violin',
                xaxis_title=None, xaxis_range=True, yaxis_title=None, yaxis_range=True, save_file=False,
                save_final=False, font_family=None, font_size=None):
        """
        Output heatmap plot of variables x and y.

        Args:
            df (dataframe): dataframe with data from heroku.
            x (list): dataframe column to plot on x axis.
            y (list): dataframe column to plot on y axis.
            pretty_text (bool, optional): prettify ticks by replacing _ with spaces and capitalising each value.
            marginal_x (str, optional): type of marginal on x axis. Can be 'histogram', 'rug', 'box', or 'violin'.
            marginal_y (str, optional): type of marginal on y axis. Can be 'histogram', 'rug', 'box', or 'violin'.
            xaxis_title (str, optional): title for x axis.
            yaxis_title (str, optional): title for y axis.
            save_file (bool, optional): flag for saving an html file with plot.
            save_final (bool, optional): flag for saving an a final figure to /figures.
            font_family (str, optional): font family to be used across the figure. None = use config value.
            font_size (int, optional): font size to be used across the figure. None = use config value.
        """
        logger.info('Creating heatmap for x={} and t={}.', x, y)
        width = tr.common.get_configs('stimulus_width')
        height = tr.common.get_configs('stimulus_height')
        x = df.loc[pp][x]
        y = df.loc[pp][y]
        # Normalize screen size
        xmin, xmax = min(x), max(x)
        for i, val in enumerate(y):
            x[i] = ((val-xmin) / (xmax-xmin))*width
        ymin, ymax = min(y), max(y)
        for i, val in enumerate(y):
            y[i] = ((val-ymin) / (ymax-ymin))*height
        t = df.loc[pp][t]
        #  prettify ticks
        if pretty_text:
            if isinstance(x, str):  # check if string
                # replace underscores with spaces
                df[x] = df[x].str.replace('_', ' ')

                # capitalise
                df[x] = df[x].str.capitalize()
            else:
                logger.error('x not a string')
            if isinstance(y, str):  # check if string
                # replace underscores with spaces
                df[y] = df[y].str.replace('_', ' ')
                # capitalise
                df[y] = df[y].str.capitalize()
            else:
                logger.error('y not a string')
            if isinstance(df.iloc[0][t], str):  # check if string
                # replace underscores with spaces
                df[t] = df[t].str.replace('_', ' ')
                # capitalise
                df[t] = df[t].str.capitalize()
        pp = str(pp)
        [go.Histogram2d(x=x[i:], y=y[i:]) for i in range(len(int(x)))]
        # build layers of animation heatmap and scatter
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y))
        fig.add_trace(go.Histogram2dContour(x=x, y=y))
        frames = [go.Frame(data=[go.Histogram2dContour(x=x[:k+1],
                                                       y=y[:k+1],
                                                       nbinsx=20,
                                                       nbinsy=20,
                                                       visible=True),
                                 go.Scatter(x=x[:k+1],
                                            y=y[:k+1],
                                            visible=True,
                                            opacity=0.9)],
                           traces=[0, 1]) for k in range(len(x))]
        fig.frames = frames
        fig.update_layout(template=self.template,
                          height=height,
                          width=width,
                          title='heatmap_scatter_animation'+' ' + id_video + ' ' + 'participant'+' '+pp,
                          xaxis_range=[0, 2*width],
                          yaxis_range=[0, 2*height],
                          updatemenus=[dict(type='buttons',
                                            buttons=[dict(label='Play',
                                                          method='animate',
                                                          args=[None,
                                                                dict(fromcurrent=True,
                                                                     transition={'duration': 10},
                                                                     frame=dict(redraw=True,
                                                                                duration=100))]),
                                                     dict(label='Pause',
                                                          method='animate',
                                                          args=[[None],
                                                                dict(fromcurrent=True,
                                                                     mode='immediate',
                                                                     transition={'duration': 10},
                                                                     frame=dict(redraw=True,
                                                                                duration=100))])])])
        # update font family
        if font_family:
            # use given value
            fig.update_layout(font=dict(family=font_family))
        else:
            # use value from config file
            fig.update_layout(font=dict(family=tr.common.get_configs('font_family')))
        # update font size
        if font_size:
            # use given value
            fig.update_layout(font=dict(size=font_size))
        else:
            # use value from config file
            fig.update_layout(font=dict(size=tr.common.get_configs('font_size')))
        # save file
        if save_file:
            self.save_plotly(fig=fig, name='heatmap_animation' + id_video+'_participant_' + pp)
        # open it in localhost instead
        else:
            # plotly.offline.plot(fig, auto_play = False)
            # TODO: error with show
            # show.fig(fig, auto_play=False)
            logger.error('Show not implemented.')

    def hist(self, df, x, nbins=None, color=None, pretty_text=False, marginal='rug', xaxis_title=None,
             yaxis_title=None, name_file=None, save_file=False, save_final=False, fig_save_width=1320,
             fig_save_height=680, font_family=None, font_size=None):
        """
        Output histogram of time of participation.

        Args:
            df (dataframe): dataframe with data from heroku.
            x (list): column names of dataframe to plot.
            nbins (int, optional): number of bins in histogram.
            color (str, optional): dataframe column to assign colour of circles.
            pretty_text (bool, optional): prettify ticks by replacing _ with spaces and capitalising each value.
            marginal (str, optional): type of marginal on x axis. Can be 'histogram', 'rug', 'box', or 'violin'.
            xaxis_title (str, optional): title for x axis.
            yaxis_title (str, optional): title for y axis.
            name_file (str, optional): name of file to save.
            save_file (bool, optional): flag for saving an html file with plot.
            save_final (bool, optional): flag for saving an a final figure to /figures.
            fig_save_width (int, optional): width of figures to be saved.
            fig_save_height (int, optional): height of figures to be saved.
            font_family (str, optional): font family to be used across the figure. None = use config value.
            font_size (int, optional): font size to be used across the figure. None = use config value.
        """
        logger.info('Creating histogram for x={}.', x)
        # using colour with multiple values to plot not supported
        if color and len(x) > 1:
            logger.error('Color property can be used only with a single variable to plot.')
            return -1
        # prettify ticks
        if pretty_text:
            for variable in x:
                # check if column contains strings
                if isinstance(df.iloc[0][variable], str):
                    # replace underscores with spaces
                    df[variable] = df[variable].str.replace('_', ' ')
                    # capitalise
                    df[variable] = df[variable].str.capitalize()
            if color and isinstance(df.iloc[0][color], str):  # check if string
                # replace underscores with spaces
                df[color] = df[color].str.replace('_', ' ')
                # capitalise
                df[color] = df[color].str.capitalize()
        # create figure
        if color:
            fig = px.histogram(df[x], nbins=nbins, marginal=marginal, color=df[color])
        else:
            fig = px.histogram(df[x], nbins=nbins, marginal=marginal)
        # ticks as numbers
        fig.update_layout(xaxis=dict(tickformat='digits'))
        # update layout
        fig.update_layout(template=self.template, xaxis_title=xaxis_title, yaxis_title=yaxis_title)
        # update font family
        if font_family:
            # use given value
            fig.update_layout(font=dict(family=font_family))
        else:
            # use value from config file
            fig.update_layout(font=dict(family=tr.common.get_configs('font_family')))
        # update font size
        if font_size:
            # use given value
            fig.update_layout(font=dict(size=font_size))
        else:
            # use value from config file
            fig.update_layout(font=dict(size=tr.common.get_configs('font_size')))
        # save file to local output folder
        if save_file:
            # build filename
            if not name_file:
                name_file = 'hist_' + '-'.join(str(val) for val in x)
            self.save_plotly(fig=fig,
                             name=name_file,
                             remove_margins=True,
                             width=fig_save_width,
                             height=fig_save_height,
                             save_final=save_final)  # also save as "final" figure
        # open it in localhost instead
        else:
            fig.show()

    def hist_stim_duration_time(self, df, time_ranges, nbins=0, font_family=None, font_size=None, name_file=None,
                                save_file=False, save_final=False, fig_save_width=1320, fig_save_height=680):
        """
        Output distribution of stimulus durations for time ranges.

        Args:
            df (dataframe): dataframe with data from heroku.
            time_ranges (dictionaries): time ranges for analysis.
            nbins (int, optional): number of bins in histogram.
            font_family (str, optional): font family to be used across the figure. None = use config value.
            font_size (int, optional): font size to be used across the figure. None = use config value.
            name_file (str, optional): name of file to save.
            save_file (bool, optional): flag for saving an html file with plot.
            save_final (bool, optional): flag for saving an a final figure to /figures.
            fig_save_width (int, optional): width of figures to be saved.
            fig_save_height (int, optional): height of figures to be saved.
        """
        logger.info('Creating histogram of stimulus durations for time ranges.')
        # columns with durations
        col_dur = df.columns[df.columns.to_series().str.contains('-dur')]
        # extract durations of stimuli
        df_dur = df[col_dur]
        df = df_dur.join(df['start'])
        df['range'] = np.nan
        # add column with labels based on time ranges
        for i, t in enumerate(time_ranges):
            for index, row in df.iterrows():
                if t['start'] <= row['start'] <= t['end']:
                    start_str = t['start'].strftime('%m-%d-%Y-%H-%M-%S')
                    end_str = t['end'].strftime('%m-%d-%Y-%H-%M-%S')
                    df.loc[index, 'range'] = start_str + ' - ' + end_str
        # drop nan
        df = df.dropna()
        # create figure
        if nbins:
            fig = px.histogram(df[col_dur], nbins=nbins, marginal='rug', color=df['range'], barmode='overlay')
        else:
            fig = px.histogram(df[col_dur], marginal='rug', color=df['range'], barmode='overlay')
        # ticks as numbers
        fig.update_layout(xaxis=dict(tickformat='digits'))
        # update layout
        fig.update_layout(template=self.template)
        # update font family
        if font_family:
            # use given value
            fig.update_layout(font=dict(family=font_family))
        else:
            # use value from config file
            fig.update_layout(font=dict(family=tr.common.get_configs('font_family')))
        # update font size
        if font_size:
            # use given value
            fig.update_layout(font=dict(size=font_size))
        else:
            # use value from config file
            fig.update_layout(font=dict(size=tr.common.get_configs('font_size')))
        # save file to local output folder
        if save_file:
            # build filename
            if not name_file:
                name_file = 'hist_stim_duration' + \
                            '-'.join(t['start'].strftime('%m.%d.%Y,%H:%M:%S') + '-' +
                                     t['end'].strftime('%m.%d.%Y,%H:%M:%S')
                                     for t in time_ranges)
            self.save_plotly(fig=fig,
                             name=name_file,
                             remove_margins=True,
                             width=fig_save_width,
                             height=fig_save_height,
                             save_final=save_final)  # also save as "final" figure
        # open it in localhost instead
        else:
            fig.show()

    def plot_kp(self, df, conf_interval=None, xaxis_title='Time (s)',
                yaxis_title='Percentage of trials with response key pressed', xaxis_range=None,
                yaxis_range=None, name_file='kp', save_file=False, save_final=False, fig_save_width=1320,
                fig_save_height=680, font_family=None, font_size=None):
        """Plot keypress data.

        Args:
            df (dataframe): dataframe with keypress data.
            conf_interval (float, optional): show confidence interval defined by argument.
            xaxis_title (str, optional): title for x axis.
            yaxis_title (str, optional): title for y axis.
            xaxis_range (list, optional): range of x axis in format [min, max].
            yaxis_range (list, optional): range of y axis in format [min, max].
            name_file (str, optional): name of file to save.
            save_file (bool, optional): flag for saving an html file with plot.
            save_final (bool, optional): flag for saving an a final figure to /figures.
            fig_save_width (int, optional): width of figures to be saved.
            fig_save_height (int, optional): height of figures to be saved.
            font_family (str, optional): font family to be used across the figure. None = use config value.
            font_size (int, optional): font size to be used across the figure. None = use config value.
        """
        logger.info('Creating visualisations of keypresses for all data.')
        # calculate times
        times = np.array(range(self.res,  df['video_length'].max() + self.res, self.res)) / 1000
        # add all data together. Must be converted to np array to add together
        kp_data = np.array([0.0] * len(times))
        for i, data in enumerate(df['kp']):
            # append zeros to match longest duration
            data = np.pad(data, (0, len(times) - len(data)), 'constant')
            # add data
            kp_data += np.array(data)
        kp_data = kp_data / (i + 1)
        # smoothen signal
        if self.smoothen_signal:
            kp_data = self.smoothen_filter(kp_data)
        # create figure
        fig = go.Figure()
        # plot keypresses
        fig = px.line(y=kp_data, x=times, title='Keypresses for all stimuli')
        # show confidence interval
        if conf_interval:
            # calculate confidence interval
            (y_lower, y_upper) = self.get_conf_interval_bounds(kp_data, conf_interval)
            # plot interval
            fig.add_trace(go.Scatter(name='Upper bound',
                                     x=times,
                                     y=y_upper,
                                     mode='lines',
                                     fillcolor='rgba(0,100,80,0.2)',
                                     line=dict(color='rgba(255,255,255,0)'),
                                     hoverinfo="skip",
                                     showlegend=False))
            fig.add_trace(go.Scatter(name='Lower bound',
                                     x=times,
                                     y=y_lower,
                                     fill='tonexty',
                                     fillcolor='rgba(0,100,80,0.2)',
                                     line=dict(color='rgba(255,255,255,0)'),
                                     hoverinfo='skip',
                                     showlegend=False))
        # define range of y axis
        if not yaxis_range:
            yaxis_range = [0, max(y_upper) if conf_interval else max(kp_data)]
        # update layout
        fig.update_layout(template=self.template,
                          xaxis_title=xaxis_title,
                          yaxis_title=yaxis_title,
                          xaxis_range=xaxis_range,
                          yaxis_range=yaxis_range)
        # update font family
        if font_family:
            # use given value
            fig.update_layout(font=dict(family=font_family))
        else:
            # use value from config file
            fig.update_layout(font=dict(family=tr.common.get_configs('font_family')))
        # update font size
        if font_size:
            # use given value
            fig.update_layout(font=dict(size=font_size))
        else:
            # use value from config file
            fig.update_layout(font=dict(size=tr.common.get_configs('font_size')))
        # save file to local output folder
        if save_file:
            self.save_plotly(fig=fig,
                             name=name_file,
                             remove_margins=True,
                             width=fig_save_width,
                             height=fig_save_height,
                             save_final=save_final)  # also save as "final" figure
        # open it in localhost instead
        else:
            fig.show()

    def plot_kp_video(self, df, stimulus, extension='mp4', conf_interval=None, vert_lines=None, events_width=3,
                      events_dash='solid', events_colour='green', events_annotations=None,
                      events_annotations_position='top right', events_annotations_font_size=20,
                      events_annotations_colour='blue', xaxis_title='Time (s)',
                      yaxis_title='Percentage of trials with response key pressed', xaxis_range=None, yaxis_range=None,
                      name_file=None, save_file=False, save_final=False, fig_save_width=1320, fig_save_height=680,
                      font_family=None, font_size=None):
        """Plot keypresses with multiple variables as a filter.

        Args:
            df (dataframe): dataframe with keypress data.
            stimulus (str): name of stimulus.
            extension (str, optional): extension of stimulus.
            conf_interval (float, optional): show confidence interval defined by argument.
            vert_lines (list, optional): list of events to draw formatted as values on x axis.
            events_width (int, optional): thickness of the vertical lines.
            events_dash (str, optional): type of the vertical lines.
            events_colour (str, optional): colour of the vertical lines.
            events_annotations (list, optional): text of annotations for the vertical lines.
            events_annotations_position (str, optional): position of annotations for the vertical lines.
            events_annotations_font_size (int, optional): font size of annotations for the vertical lines.
            events_annotations_colour (str, optional): colour of annotations for the vertical lines.
            xaxis_title (str, optional): title for x axis.
            yaxis_title (str, optional): title for y axis.
            xaxis_range (list, optional): range of x axis in format [min, max].
            yaxis_range (list, optional): range of y axis in format [min, max].
            name_file (str, optional): name of file to save.
            save_file (bool, optional): flag for saving an html file with plot.
            save_final (bool, optional): flag for saving an a final figure to /figures.
            fig_save_width (int, optional): width of figures to be saved.
            fig_save_height (int, optional): height of figures to be saved.
            font_family (str, optional): font family to be used across the figure. None = use config value.
            font_size (int, optional): font size to be used across the figure. None = use config value.
        """
        # extract video length
        video_len = df.loc[stimulus]['video_length']
        # calculate times
        times = np.array(range(self.res, video_len + self.res, self.res)) / 1000
        # keypress data
        kp_data = df.loc[stimulus]['kp']
        # smoothen signal
        if self.smoothen_signal:
            kp_data = self.smoothen_filter(kp_data)
        # plot keypresses
        fig = px.line(y=df.loc[stimulus]['kp'], x=times, title='Keypresses for stimulus ' + stimulus)
        # show confidence interval
        if conf_interval:
            # calculate confidence interval
            (y_lower, y_upper) = self.get_conf_interval_bounds(kp_data, conf_interval)
            # plot interval
            fig.add_trace(go.Scatter(name='Upper bound',
                                     x=times,
                                     y=y_upper,
                                     mode='lines',
                                     fillcolor='rgba(0,100,80,0.2)',
                                     line=dict(color='rgba(255,255,255,0)'),
                                     hoverinfo="skip",
                                     showlegend=False))
            fig.add_trace(go.Scatter(name='Lower bound',
                                     x=times,
                                     y=y_lower,
                                     fill='tonexty',
                                     fillcolor='rgba(0,100,80,0.2)',
                                     line=dict(color='rgba(255,255,255,0)'),
                                     hoverinfo="skip",
                                     showlegend=False))
        # draw vertical lines with annotations
        if vert_lines:
            for line, annotation in zip(vert_lines, events_annotations):
                fig.add_vline(
                    x=line,
                    line_width=events_width,
                    line_dash=events_dash,
                    line_color=events_colour,
                    annotation_text=annotation,
                    annotation_position=events_annotations_position,
                    annotation_font_size=events_annotations_font_size,
                    annotation_font_color=events_annotations_colour)
        # define range of y axis
        if not yaxis_range:
            yaxis_range = [0, max(y_upper) if conf_interval else max(kp_data)]
        # update layout
        fig.update_layout(template=self.template,
                          xaxis_title=xaxis_title,
                          yaxis_title=yaxis_title,
                          xaxis_range=xaxis_range,
                          yaxis_range=yaxis_range)
        # update font family
        if font_family:
            # use given value
            fig.update_layout(font=dict(family=font_family))
        else:
            # use value from config file
            fig.update_layout(font=dict(family=tr.common.get_configs('font_family')))
        # update font size
        if font_size:
            # use given value
            fig.update_layout(font=dict(size=font_size))
        else:
            # use value from config file
            fig.update_layout(font=dict(size=tr.common.get_configs('font_size')))
        # save file to local output folder
        if save_file:
            # build filename
            if not name_file:
                name_file = 'kp_' + stimulus
            self.save_plotly(fig=fig,
                             name=name_file,
                             remove_margins=True,
                             width=fig_save_width,
                             height=fig_save_height,
                             save_final=save_final)  # also save as "final" figure
        # open it in localhost instead
        else:
            fig.show()

    def plot_kp_video_pp(self, df, dt, stimulus, pp, conf_interval=None, trendline=None, xaxis_title='Time (s)',
                         yaxis_title='response key pressed', xaxis_range=None, yaxis_range=None, name_file=None,
                         save_file=False, save_final=False, fig_save_width=1320, fig_save_height=680, font_family=None,
                         font_size=None):
        """Plot keypresses data of one stimulus for 1 participant.

        Args:
            df (dataframe): dataframe with stimulus data.
            dt (dataframe): dataframe with keypress data.
            stimulus (str): name of stimulus.
            pp (str): ID of participant.
            conf_interval (float, optional): show confidence interval defined by argument.
            trendline (None, optional): Description
            xaxis_title (str, optional): title for x axis.
            yaxis_title (str, optional): title for y axis.
            xaxis_range (list, optional): range of x axis in format [min, max].
            yaxis_range (list, optional): range of y axis in format [min, max].
            name_file (str, optional): name of file to save.
            save_file (bool, optional): flag for saving an html file with plot.
            save_final (bool, optional): flag for saving an a final figure to /figures.
            fig_save_width (int, optional): width of figures to be saved.
            fig_save_height (int, optional): height of figures to be saved.
            font_family (str, optional): font family to be used across the figure. None = use config value.
            font_size (int, optional): font size to be used across the figure. None = use config value.
        """
        # todo: implement for 1 pp
        # extract video length
        video_len = df.loc[stimulus]['video_length']
        # calculate times
        times = np.array(range(self.res, video_len + self.res, self.res)) / 1000
        # keypress data
        kp_data_time = dt.loc[pp][stimulus + '-rt-0']
        kp_ar = np.array(kp_data_time)
        kp_data = np.where(kp_ar > 0, 1, 0)
        # smoothen signal
        if self.smoothen_signal:
            kp_data = self.smoothen_filter(kp_data)
        kp_data_time = kp_ar/100
        # plot keypresses
        fig = px.line(x=kp_data_time,
                      y=kp_data,
                      # animation_frame=kp_data_time,
                      title='Keypresses for stimulus ' + stimulus + ' for participant ' + pp)
        # update layout
        fig.update_layout(template=self.template,
                          xaxis_title=xaxis_title,
                          yaxis_title=yaxis_title,
                          xaxis_range=[0, len(times)],
                          yaxis_range=yaxis_range)
        # update font family
        if font_family:
            # use given value
            fig.update_layout(font=dict(family=font_family))
        else:
            # use value from config file
            fig.update_layout(font=dict(family=tr.common.get_configs('font_family')))
        # update font size
        if font_size:
            # use given value
            fig.update_layout(font=dict(size=font_size))
        else:
            # use value from config file
            fig.update_layout(font=dict(size=tr.common.get_configs('font_size')))
        # save file to local output folder
        if save_file:
            # build filename
            if not name_file:
                name_file = 'kp_' + stimulus + '_' + pp
            self.save_plotly(fig=fig,
                             name=name_file,
                             remove_margins=True,
                             width=fig_save_width,
                             height=fig_save_height,
                             save_final=save_final)  # also save as "final" figure
        # open it in localhost instead
        else:
            fig.show()

    def plot_kp_animate(self, df, stimulus, pp='all', conf_interval=None, xaxis_title='Time (s)',
                        yaxis_title='Percentage of trials with response key pressed', xaxis_range=None,
                        yaxis_range=None, name_file=None, save_file=False, save_final=False, fig_save_width=1320,
                        fig_save_height=680, font_family=None, font_size=None):
        """Animation of keypress data.

        Args:
           df (dataframe): dataframe with stimulus data.
            stimulus (str): name of stimulus.
            pp (str): ID of participant.
            conf_interval (None, optional): show confidence interval.
            xaxis_title (str, optional): title for x axis.
            yaxis_title (str, optional): title for y axis.
            xaxis_range (list, optional): range of x axis in format [min, max].
            yaxis_range (list, optional): range of y axis in format [min, max].
            name_file (str, optional): name of file to save.
            save_file (bool, optional): flag for saving an html file with plot.
            save_final (bool, optional): flag for saving an a final figure to /figures.
            fig_save_width (int, optional): width of figures to be saved.
            fig_save_height (int, optional): height of figures to be saved.
            font_family (str, optional): font family to be used across the figure. None = use config value.
            font_size (int, optional): font size to be used across the figure. None = use config value.
        """
        # extract video length
        video_len = df.loc[stimulus]['video_length']
        # calculate times
        times = np.array(range(self.res, video_len + self.res, self.res)) / 1000
        # keypress data
        kp_data = df.loc[stimulus]['kp']
        # smoothen signal
        if self.smoothen_signal:
            kp_data = self.smoothen_filter(kp_data)
        # plot keypresses
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=times, y=kp_data))
        frames = [go.Frame(data=[go.Scatter(x=times[:k+1],
                                            y=kp_data[:k+1],
                                            visible=True,
                                            opacity=0.9)],
                           traces=[0]) for k in range(len(times))]
        fig.frames = frames
        fig.update_layout(template=self.template,
                          title='Keypresses for stimulus ' + stimulus,
                          updatemenus=[dict(type='buttons',
                                            buttons=[dict(label='Play',
                                                          method='animate',
                                                          args=[None,
                                                                dict(fromcurrent=True,
                                                                     transition={'duration': 10},
                                                                     frame=dict(redraw=True,
                                                                                duration=100))]),
                                                     dict(label='Pause',
                                                          method='animate',
                                                          args=[[None],
                                                                dict(fromcurrent=True,
                                                                     mode='immediate',
                                                                     transition={'duration': 10},
                                                                     frame=dict(redraw=True, duration=100))])])])
        if conf_interval:
            # calculate confidence interval
            (y_lower, y_upper) = self.get_conf_interval_bounds(kp_data, conf_interval)
            # plot interval
            fig.add_trace(go.Scatter(name='Upper bound',
                                     x=times,
                                     y=y_upper,
                                     mode='lines',
                                     fillcolor='rgba(0,100,80,0.2)',
                                     line=dict(color='rgba(255,255,255,0)'),
                                     hoverinfo='skip',
                                     showlegend=False))
            fig.add_trace(go.Scatter(name='Lower bound',
                                     x=times,
                                     y=y_lower,
                                     fill='tonexty',
                                     fillcolor='rgba(0,100,80,0.2)',
                                     line=dict(color='rgba(255,255,255,0)'),
                                     hoverinfo='skip',
                                     showlegend=False))
        fig.update_layout(template=self.template,
                          xaxis_title=xaxis_title,
                          yaxis_title=yaxis_title,
                          xaxis_range=xaxis_range,
                          yaxis_range=yaxis_range)
        # update font family
        if font_family:
            # use given value
            fig.update_layout(font=dict(family=font_family))
        else:
            # use value from config file
            fig.update_layout(font=dict(family=tr.common.get_configs('font_family')))
        # update font size
        if font_size:
            # use given value
            fig.update_layout(font=dict(size=font_size))
        else:
            # use value from config file
            fig.update_layout(font=dict(size=tr.common.get_configs('font_size')))
        # save file to local output folder
        if save_file:
            # build filename
            if not name_file:
                name_file = 'kp_animation' + stimulus
            self.save_plotly(fig=fig,
                             name=name_file,
                             remove_margins=True,
                             width=fig_save_width,
                             height=fig_save_height,
                             save_final=save_final)  # also save as "final" figure
        # open it in localhost instead
        else:
            fig.show()

    def plot_video_data(self, df, stimulus, cols, conf_interval=None, xaxis_title='Time (s)',
                        yaxis_title='Percentage of trials with response key pressed', xaxis_range=None,
                        yaxis_range=None, name_file=None, save_file=False, save_final=False, fig_save_width=1320,
                        fig_save_height=680, font_family=None, font_size=None):
        """Plot keypresses with multiple variables as a filter.

        Args:
            df (dataframe): dataframe with keypress data.
            stimulus (str): name of stimulus.
            cols: columns of which to plot
            conf_interval (float, optional): show confidence interval defined by argument.
            xaxis_title (str, optional): title for x axis.
            yaxis_title (str, optional): title for y axis.
            xaxis_range (list, optional): range of x axis in format [min, max].
            yaxis_range (list, optional): range of y axis in format [min, max].
            name_file (str, optional): name of file to save.
            save_file (bool, optional): flag for saving an html file with plot.
            save_final (bool, optional): flag for saving an a final figure to /figures.
            fig_save_width (int, optional): width of figures to be saved.
            fig_save_height (int, optional): height of figures to be saved.
            font_family (str, optional): font family to be used across the figure. None = use config value.
            font_size (int, optional): font size to be used across the figure. None = use config value.
        """
        # plotly figure to make plots in
        fig = subplots.make_subplots(rows=1, cols=1, shared_xaxes=True)
        # plot for all videos
        # buttons = list([dict(label='All',
        #                      method='update',
        #                      args=[{'visible': [True] * df.shape[0]},
        #                            {'title': 'Keypresses for individual stimuli',
        #                             'showlegend': True}])])

        for col in cols:
            video_len = df.loc[stimulus]['video_length']
            # calculate times
            times = np.array(range(self.res, video_len + self.res, self.res)) / 1000
            # keypress data
            kp_data = df.loc[stimulus][col]
            if type(kp_data) is not list:
                kp_data = ast.literal_eval(kp_data)
            # smoothen signal
            if self.smoothen_signal:
                kp_data = self.smoothen_filter(kp_data)
            # plot keypresses
            fig.add_trace(go.Scatter(y=kp_data,
                                     mode='lines',
                                     x=times,
                                     name=col),
                          row=1,
                          col=1)
        # update layout
        fig.update_layout(template=self.template,
                          title=stimulus,
                          xaxis_title=xaxis_title,
                          yaxis_title=yaxis_title,
                          xaxis_range=xaxis_range,
                          yaxis_range=yaxis_range)
        # update font family
        if font_family:
            # use given value
            fig.update_layout(font=dict(family=font_family))
        else:
            # use value from config file
            fig.update_layout(font=dict(family=tr.common.get_configs('font_family')))
        # update font size
        if font_size:
            # use given value
            fig.update_layout(font=dict(size=font_size))
        else:
            # use value from config file
            fig.update_layout(font=dict(size=tr.common.get_configs('font_size')))
        # save file to local output folder
        if save_file:
            # build filename
            if not name_file:
                name_file = 'video_data_' + stimulus
            self.save_plotly(fig=fig,
                             name=name_file,
                             remove_margins=True,
                             width=fig_save_width,
                             height=fig_save_height,
                             save_final=save_final)  # also save as "final" figure
        # open it in localhost instead
        else:
            fig.show()

    def plot_kp_videos(self, df, events=None, events_width=1, events_dash='dot', events_colour='black',
                       events_annotations_font_size=20, events_annotations_colour='black', xaxis_title='Time (s)',
                       yaxis_title='Percentage of trials with response key pressed', xaxis_range=None,
                       yaxis_range=None, save_file=False, save_final=False, fig_save_width=1320, fig_save_height=680,
                       show_menu=False, show_title=True, name_file='kp_videos', font_family=None, font_size=None):
        """Plot keypresses with multiple variables as a filter.

        Args:
            df (dataframe): dataframe with keypress data.
            events (list, optional): list of events to draw formatted as values on x axis.
            events_width (int, optional): thickness of the vertical lines.
            events_dash (str, optional): type of the vertical lines.
            events_colour (str, optional): colour of the vertical lines.
            events_annotations_font_size (int, optional): font size of annotations for the vertical lines.
            events_annotations_colour (str, optional): colour of annotations for the vertical lines.
            xaxis_title (str, optional): title for x axis.
            yaxis_title (str, optional): title for y axis.
            xaxis_range (list, optional): range of x axis in format [min, max].
            yaxis_range (list, optional): range of y axis in format [min, max].
            save_file (bool, optional): flag for saving an html file with plot.
            save_final (bool, optional): flag for saving an a final figure to /figures.
            fig_save_width (int, optional): width of figures to be saved.
            fig_save_height (int, optional): height of figures to be saved.
            show_menu (bool, optional): show menu on top left with variables to select for plotting.
            show_title (bool, optional): show title on top of figure.
            name_file (str, optional): name of file to save.
            font_family (str, optional): font family to be used across the figure. None = use config value.
            font_size (int, optional): font size to be used across the figure. None = use config value.
        """
        # calculate times
        times = np.array(range(self.res, df['video_length'].max() + self.res, self.res)) / 1000
        # plotly
        fig = subplots.make_subplots(rows=1, cols=1, shared_xaxes=True)
        # plot for all videos
        for index, row in df.iterrows():
            values = row['kp']  # keypress data
            # smoothen signal
            if self.smoothen_signal:
                values = self.smoothen_filter(values)
            fig.add_trace(go.Scatter(y=values,
                                     mode='lines',
                                     x=times,
                                     name=os.path.splitext(index)[0]),
                          row=1,
                          col=1)
        # count lines to calculate increase in coordinates of drawing
        counter_lines = 0
        # draw lines with annotations for events
        if events:
            for event in events:
                # draw start
                fig.add_shape(type='line',
                              x0=event['start'],
                              y0=0,
                              x1=event['start'],
                              y1=yaxis_range[1] - counter_lines * 1.8 - 1,
                              line=dict(color=events_colour,
                                        dash='dot',
                                        width=events_width))
                # draw finish
                fig.add_shape(type='line',
                              x0=event['end'],
                              y0=0,
                              x1=event['end'],
                              y1=yaxis_range[1] - counter_lines * 1.8 - 1,
                              line=dict(color=events_colour,
                                        dash=events_dash,
                                        width=events_width))
                # draw horizontal line
                fig.add_annotation(ax=event['start'],
                                   axref='x',
                                   ay=yaxis_range[1] - counter_lines * 1.8 - 1,
                                   ayref='y',
                                   x=event['end'],
                                   arrowcolor='black',
                                   xref='x',
                                   y=yaxis_range[1] - counter_lines * 1.8 - 1,
                                   yref='y',
                                   arrowwidth=events_width,
                                   arrowside='end+start',
                                   arrowsize=1,
                                   arrowhead=2)
                # draw text label
                fig.add_annotation(text=event['annotation'],
                                   # xref='paper', yref='paper',
                                   x=(event['end'] + event['start']) / 2,
                                   y=yaxis_range[1] - counter_lines * 1.8,  # use ylim value and draw lower
                                   showarrow=False,
                                   font=dict(size=events_annotations_font_size,
                                             color=events_annotations_colour))
                # increase counter of lines drawn
                counter_lines = counter_lines + 1
        # buttons with the names of stimuli
        buttons = list([dict(label='All',
                             method='update',
                             args=[{'visible': [True] * df.shape[0]},
                                   {'title': 'Keypresses for individual stimuli', 'showlegend': True}])])
        # show menu with selection of variable to plot
        if show_menu:
            # counter for traversing through stimuli
            counter_rows = 0
            # go over extracted videos
            for index, row in df.iterrows():
                visibility = [[counter_rows == j] for j in range(df.shape[0])]
                visibility = [item for sublist in visibility for item in sublist]
                button = dict(label=os.path.splitext(index)[0],
                              method='update',
                              args=[{'visible': visibility},
                                    {'title': os.path.splitext(index)[0]}])
                buttons.append(button)
                counter_rows = counter_rows + 1
            updatemenus = [dict(x=-0.15, buttons=buttons, showactive=True)]
            fig['layout']['updatemenus'] = updatemenus
        # update layout
        if show_title:
            fig['layout']['title'] = 'Keypresses for individual stimuli'
        # update layout
        fig.update_layout(template=self.template,
                          xaxis_title=xaxis_title,
                          yaxis_title=yaxis_title,
                          xaxis_range=xaxis_range,
                          yaxis_range=yaxis_range)
        # update font family
        if font_family:
            # use given value
            fig.update_layout(font=dict(family=font_family))
        else:
            # use value from config file
            fig.update_layout(font=dict(family=tr.common.get_configs('font_family')))
        # update font size
        if font_size:
            # use given value
            fig.update_layout(font=dict(size=font_size))
        else:
            # use value from config file
            fig.update_layout(font=dict(size=tr.common.get_configs('font_size')))
        # save file to local output folder
        if save_file:
            self.save_plotly(fig=fig,
                             name=name_file,
                             remove_margins=True,
                             width=fig_save_width,
                             height=fig_save_height,
                             save_final=save_final)  # also save as "final" figure
        # open it in localhost instead
        else:
            fig.show()


    def plot_kp_slider_videos(self, df, stim, y: list, y_legend=None, x=None, events=None, events_width=1, events_dash='dot',
                              events_colour='black', events_annotations_font_size=20,
                              events_annotations_colour='black', xaxis_kp_title='Time (s)',
                              yaxis_kp_title='Percentage of trials with response key pressed',
                              xaxis_kp_range=None, yaxis_kp_range=None, yaxis_slider_range=None, stacked=False, pretty_text=False,
                              orientation='v', xaxis_slider_title='Stimulus', yaxis_slider_show=False,
                              yaxis_slider_title=None, show_text_labels=False, xaxis_ticklabels_slider_show=True,
                              yaxis_ticklabels_slider_show=False, name_file='kp_videos_sliders', save_file=False,
                              save_final=False, fig_save_width=1600, fig_save_height=1280, legend_x=0.7, legend_y=0.95,
                              font_family=None, font_size=None, 
                              # ttest_results_file=None, 
                              # ttest_marker='circle',
                              # ttest_marker_size=3,  
                              # ttest_marker_colour='black', 
                              # ttest_annotations_font_size=10,
                              # ttest_annotations_colour='black', 
                              anova_results_file=None,
                              nonparametric_results_file=None, 
                              within_marker = 'diamond',
                              within_marker_size = 3,
                              within_marker_colour = 'blue',
                              between_marker = 'diamond',
                              between_marker_size = 3,
                              between_marker_colour = 'green',
                              interaction_marker = 'diamond',
                              interaction_marker_size = 3,
                              interaction_marker_colour = 'red',
                              within_ego0_marker='square',
                              within_ego0_marker_size=4,
                              within_ego0_marker_colour='blue',
                              within_ego1_marker='diamond',
                              within_ego1_marker_size=4,
                              within_ego1_marker_colour='blue',
                              within_target0_marker='square',
                              within_target0_marker_size=4,
                              within_target0_marker_colour='green',
                              within_target1_marker='diamond',
                              within_target1_marker_size=4,
                              within_target1_marker_colour='green',
                              comp_mixed_marker='diamond',
                              comp_mixed_marker_size=4,
                              comp_mixed_marker_colour='purple',
                              comp_simple_marker='diamond',
                              comp_simple_marker_size=4,
                              comp_simple_marker_colour='red',
                              anova_annotations_font_size=10,
                              anova_annotations_colour='black', ttest_anova_row_height=0.5, yaxis_step=10):
        """Plot keypresses with multiple variables as a filter and slider questions for the stimuli.

        Args:
            df (dataframe): dataframe with stimuli data.
            y (list): column names of dataframe to plot.
            y_legend (list, optional): names for variables to be shown in the legend.
            x (list): values in index of dataframe to plot for. If no value is given, the index of df is used.
            events (list, optional): list of events to draw formatted as values on x axis.
            events_width (int, optional): thickness of the vertical lines.
            events_dash (str, optional): type of the vertical lines.
            events_colour (str, optional): colour of the vertical lines.
            events_annotations_font_size (int, optional): font size of annotations for the vertical lines.
            events_annotations_colour (str, optional): colour of annotations for the vertical lines.
            xaxis_kp_title (str, optional): title for x axis. for the keypress plot
            yaxis_kp_title (str, optional): title for y axis. for the keypress plot
            xaxis_kp_range (None, optional): range of x axis in format [min, max] for the keypress plot.
            yaxis_kp_range (None, optional): range of x axis in format [min, max] for the keypress plot.
            stacked (bool, optional): show as stacked chart.
            pretty_text (bool, optional): prettify ticks by replacing _ with spaces and capitalising each value.
            orientation (str, optional): orientation of bars. v=vertical, h=horizontal.
            xaxis_slider_title (None, optional): title for x axis. for the slider data plot.
            yaxis_slider_show (bool, optional): show y axis or not.
            yaxis_slider_title (None, optional): title for y axis. for the slider data plot.
            show_text_labels (bool, optional): output automatically positioned text labels.
            xaxis_ticklabels_slider_show (bool, optional): show tick labels for slider plot.
            yaxis_ticklabels_slider_show (bool, optional): show tick labels for slider plot.
            name_file (str, optional): name of file to save.
            save_file (bool, optional): flag for saving an html file with plot.
            save_final (bool, optional): flag for saving an a final figure to /figures.
            fig_save_width (int, optional): width of figures to be saved.
            fig_save_height (int, optional): height of figures to be saved.
            legend_x (float, optional): location of legend, percentage of x axis.
            legend_y (float, optional): location of legend, percentage of y axis.
            font_family (str, optional): font family to be used across the figure. None = use config value.
            font_size (int, optional): font size to be used across the figure. None = use config value.
            ttest_signals (list, optional): signals to compare with ttest. None = do not compare.
            ttest_marker (str, optional): symbol of markers for the ttest.
            ttest_marker_size (int, optional): size of markers for the ttest.
            ttest_marker_colour (str, optional): colour of markers for the ttest.
            ttest_annotations_font_size (int, optional): font size of annotations for ttest.
            ttest_annotations_colour (str, optional): colour of annotations for ttest.
            anova_signals (dict, optional): signals to compare with ANOVA. None = do not compare.
            anova_marker (str, optional): symbol of markers for the ANOVA.
            anova_marker_size (int, optional): size of markers for the ANOVA.
            anova_marker_colour (str, optional): colour of markers for the ANOVA.
            anova_annotations_font_size (int, optional): font size of annotations for ANOVA.
            anova_annotations_colour (str, optional): colour of annotations for ANOVA.
            ttest_anova_row_height (int, optional): height of row of ttest/anova markers.
            yaxis_step (int): step between ticks on y axis.
        """
        logger.info('Creating figure keypress and slider data for {}.', df.index.tolist())
        logger.info(f"Creating figure for stimulus group: {stim}")
        # calculate times
        times = np.array(range(self.res, df['video_length'].max() + self.res, self.res)) / 1000
        # plotly
        fig = subplots.make_subplots(rows=1,
                                     cols=2,
                                     column_widths=[0.8, 0.2],
                                     subplot_titles=('Mean keypress values', 'Responses to sliders'),
                                     specs=[[{}, {}]],
                                     horizontal_spacing=0.00,
                                     shared_xaxes=False)
        # # adjust ylim, if ttest results need to be plotted
        # if ttest_signals:
        #     # assume one row takes ttest_anova_row_height on y axis
        #     yaxis_kp_range[0] = round(yaxis_kp_range[0] - len(ttest_signals) * ttest_anova_row_height - ttest_anova_row_height)  # noqa: E501
        # # adjust ylim, if anova results need to be plotted
        # if anova_signals:
        #     # assume one row takes ttest_anova_row_height on y axis
        #     yaxis_kp_range[0] = round(yaxis_kp_range[0] - len(anova_signals) * ttest_anova_row_height - ttest_anova_row_height)  # noqa: E501
        # Adjust ylim based on t-test and ANOVA annotations
        # if ttest_results_file:
        #     yaxis_kp_range[0] -= ttest_anova_row_height
        if anova_results_file:
            yaxis_kp_range[0] -= ttest_anova_row_height
        # plot keypress data
        for index, row in df.iterrows():
            values = row['kp']  # keypress data
            # smoothen signal
            if self.smoothen_signal:
                values = self.smoothen_filter(values)
            # plot signal
            fig.add_trace(go.Scatter(y=values,
                                     mode='lines',
                                     x=times,
                                     name=os.path.splitext(index)[0]),
                          row=1,
                          col=1)
        # draw events
        self.draw_events(fig=fig,
                         yaxis_range=yaxis_kp_range,
                         events=events,
                         events_width=events_width,
                         events_dash=events_dash,
                         events_colour=events_colour,
                         events_annotations_font_size=events_annotations_font_size,
                         events_annotations_colour=events_annotations_colour)
        # update axis
        fig.update_xaxes(title_text=xaxis_kp_title, range=xaxis_kp_range, row=1, col=1)
        fig.update_yaxes(title_text=yaxis_kp_title, range=yaxis_kp_range, row=1, col=1)
        # prettify text
        if pretty_text:
            for variable in y:
                # check if column contains strings
                if isinstance(df.iloc[0][variable], str):
                    # replace underscores with spaces
                    df[variable] = df[variable].str.replace('_', ' ')
                    # capitalise
                    df[variable] = df[variable].str.capitalize()
        # Plot slider data
        # use index of df if none is given
        if not x:
            x = df.index
        # go over variables to plot
        for variable in range(len(y)):
            # showing text labels
            if show_text_labels:
                text = df[y[variable]]
            else:
                text = None
            # custom labels for legend
            if y_legend:
                name = y_legend[variable]
            else:
                name = y[variable]
            # plot variable
            fig.add_trace(go.Bar(x=x,
                                 y=df[y[variable]],
                                 name=name,
                                 orientation=orientation,
                                 text=text,
                                 textposition='auto'),
                          row=1,
                          col=2)
        # draw ttest and anova rows
        # self.draw_anova_from_precomputed_results(
        #                                             fig=fig,
        #                                             times=times,
        #                                             anova_results_path=os.path.join(self.anova_dir, f'batch_{stim}_keypress_data_rank_anova_results.csv'),
        #                                             yaxis_range=yaxis_kp_range,
        #                                             yaxis_step=yaxis_step,
        #                                             anova_marker=anova_marker,
        #                                             anova_marker_size=anova_marker_size,
        #                                             anova_marker_colour=anova_marker_colour,
        #                                             anova_annotations_font_size=anova_annotations_font_size,
        #                                             anova_annotations_colour=anova_annotations_colour
        #                                         )

        self.draw_ttest_anova_from_files(fig=fig,
                                          times=times,
                                          name_file=name_file,
                                          stim=stim,
                                          yaxis_range=yaxis_kp_range,
                                          yaxis_step=yaxis_step,
                                          # ttest_results_file=None, 
                                          # ttest_marker=None,
                                          # ttest_marker_size=None,
                                          # ttest_marker_colour=ttest_marker_colour,
                                          # ttest_annotations_font_size=ttest_annotations_font_size,
                                          # ttest_annotations_colour=ttest_annotations_colour,
                                          anova_results_file=os.path.join(tr.settings.output_dir, 'statistics', f"batch_{stim}_keypress_data_rank_anova_results.csv"),
                                          nonparametric_results_file=os.path.join(tr.settings.output_dir, 'statistics', f"batch_{stim}_keypress_data_nonparametric_results.csv"),
                                          within_marker='diamond', within_marker_size=4, within_marker_colour='blue',
                                          between_marker='diamond', between_marker_size=4, between_marker_colour='green',
                                          interaction_marker='diamond', interaction_marker_size=4, interaction_marker_colour='red',
                                          within_ego0_marker='square',
                                          within_ego0_marker_size=4,
                                          within_ego0_marker_colour='blue',
                                          within_ego1_marker='diamond',
                                          within_ego1_marker_size=4,
                                          within_ego1_marker_colour='blue',
                                          within_target0_marker='square',
                                          within_target0_marker_size=4,
                                          within_target0_marker_colour='green',
                                          within_target1_marker='diamond',
                                          within_target1_marker_size=4,
                                          within_target1_marker_colour='green',
                                          comp_mixed_marker='diamond',
                                          comp_mixed_marker_size=4,
                                          comp_mixed_marker_colour='purple',
                                          comp_simple_marker='diamond',
                                          comp_simple_marker_size=4,
                                          comp_simple_marker_colour='red',

                                          # anova_marker_size=anova_marker_size,
                                          # anova_marker_colour=anova_marker_colour,
                                          anova_annotations_font_size=anova_annotations_font_size,
                                          anova_annotations_colour=anova_annotations_colour,
                                          ttest_anova_row_height=ttest_anova_row_height)
        # update slider axis
        fig.update_xaxes(title_text=xaxis_slider_title, row=1, col=2)
        fig.update_yaxes(title_text=yaxis_slider_title, range=yaxis_slider_range, visible=yaxis_slider_show, row=1, col=2)
        fig.update_xaxes(showticklabels=xaxis_ticklabels_slider_show, row=1, col=2)
        fig.update_yaxes(showticklabels=yaxis_ticklabels_slider_show, row=1, col=2)
        # update template
        fig.update_layout(template=self.template)
        # format text labels
        if show_text_labels:
            fig.update_traces(texttemplate='%{text:.2f}')
        # stacked bar chart
        if stacked:
            fig.update_layout(barmode='stack')
        # legend
        fig.update_layout(legend=dict(x=legend_x, y=legend_y))
        # update font family
        if font_family:
            # use given value
            fig.update_layout(font=dict(family=font_family))
        else:
            # use value from config file
            fig.update_layout(font=dict(family=tr.common.get_configs('font_family')))
        # update font size
        if font_size:
            # use given value
            fig.update_layout(font=dict(size=font_size))
        else:
            # use value from config file
            fig.update_layout(font=dict(size=tr.common.get_configs('font_size')))
        # save file to local output folder
        if save_file:
            self.save_plotly(fig=fig,
                             name=name_file,
                             remove_margins=True,
                             width=fig_save_width,
                             height=fig_save_height,
                             save_final=save_final)  # also save as "final" figure
        # open it in localhost instead
        else:
            fig.show()

    def plot_kp_variable(self, df, variable, y_legend=None, values=None, xaxis_title='Time (s)',
                         yaxis_title='Percentage of trials with response key pressed', xaxis_range=None,
                         yaxis_range=None, show_menu=False, show_title=True, name_file=None, save_file=False,
                         save_final=False, fig_save_width=1320, fig_save_height=680, legend_x=0, legend_y=0,
                         font_family=None, font_size=None, events=None, events_width=1, events_dash='dot',
                         events_colour='black', events_annotations_font_size=20, events_annotations_colour='black',
                         ttest_signals=None, ttest_marker='circle', ttest_marker_size=3,  ttest_marker_colour='black',
                         ttest_annotations_font_size=10, ttest_annotations_colour='black', anova_signals=None,
                         anova_marker='cross', anova_marker_size=3, anova_marker_colour='black',
                         anova_annotations_font_size=10, anova_annotations_colour='black', ttest_anova_row_height=0.5,
                         yaxis_step=10):
        """Plot figures of values of a certain variable.

        Args:
            df (dataframe): dataframe with keypress data.
            variable (str): variable to plot.
            y_legend (list, optional): names for variables to be shown in the legend.
            values (list, optional): values of variable to plot. If None, all values are plotted.
            xaxis_title (str, optional): title for x axis.
            yaxis_title (str, optional): title for y axis.
            xaxis_range (list, optional): range of x axis in format [min, max].
            yaxis_range (list, optional): range of y axis in format [min, max].
            show_menu (bool, optional): show menu on top left with variables to select for plotting.
            show_title (bool, optional): show title on top of figure.
            name_file (str, optional): name of file to save.
            save_file (bool, optional): flag for saving an html file with plot.
            save_final (bool, optional): flag for saving an a final figure to /figures.
            fig_save_width (int, optional): width of figures to be saved.
            fig_save_height (int, optional): height of figures to be saved.
            legend_x (float, optional): location of legend, percentage of x axis. 0 = use default value.
            legend_y (float, optional): location of legend, percentage of y axis. 0 = use default value.
            font_family (str, optional): font family to be used across the figure. None = use config value.
            font_size (int, optional): font size to be used across the figure. None = use config value.
            events (list, optional): list of events to draw formatted as values on x axis.
            events_width (int, optional): thickness of the vertical lines.
            events_dash (str, optional): type of the vertical lines.
            events_colour (str, optional): colour of the vertical lines.
            events_annotations_font_size (int, optional): font size of annotations for the vertical lines.
            events_annotations_colour (str, optional): colour of annotations for the vertical lines.
            ttest_signals (list, optional): signals to compare with ttest. None = do not compare.
            ttest_marker (str, optional): symbol of markers for the ttest.
            ttest_marker_size (int, optional): size of markers for the ttest.
            ttest_marker_colour (str, optional): colour of markers for the ttest.
            ttest_annotations_font_size (int, optional): font size of annotations for ttest.
            ttest_annotations_colour (str, optional): colour of annotations for ttest.
            anova_signals (dict, optional): signals to compare with ANOVA. None = do not compare.
            anova_marker (str, optional): symbol of markers for the ANOVA.
            anova_marker_size (int, optional): size of markers for the ANOVA.
            anova_marker_colour (str, optional): colour of markers for the ANOVA.
            anova_annotations_font_size (int, optional): font size of annotations for ANOVA.
            anova_annotations_colour (str, optional): colour of annotations for ANOVA.
            ttest_anova_row_height (int, optional): height of row of ttest/anova markers.
            yaxis_step (int): step between ticks on y axis.
            ttest_anova_row_height (int, optional): height of row of ttest/anova markers.
            yaxis_step (int): step between ticks on y axis.
        """
        logger.info('Creating visualisation of keypresses based on values {} of variable {}.', values, variable)
        # calculate times
        times = np.array(range(self.res, df['video_length'].max() + self.res, self.res)) / 1000
        # adjust ylim, if ttest results need to be plotted
        if ttest_signals:
            # assume one row takes ttest_anova_row_height on y axis
            yaxis_range[0] = round(yaxis_range[0] - len(ttest_signals) * ttest_anova_row_height - ttest_anova_row_height)  # noqa: E501
        # adjust ylim, if anova results need to be plotted
        if anova_signals:
            # assume one row takes ttest_anova_row_height on y axis
            yaxis_range[0] = round(yaxis_range[0] - len(anova_signals) * ttest_anova_row_height - ttest_anova_row_height)  # noqa: E501
        # if no values specified, plot value
        if not values:
            values = df[variable].unique()
        # extract data for values
        extracted_data = []
        for value in values:
            kp_data = np.array([0.0] * len(times))
            # non-nan value (provide as np.nan)
            if not pd.isnull(value):
                df_f = df[df[variable] == value]
            # nan value
            else:
                df_f = df[df[variable].isnull()]
            # go over extracted videos
            for index, row in df_f.iterrows():
                data_row = np.array(row['kp'])
                # append zeros to match longest duration
                data_row = np.pad(data_row, (0, len(times) - len(data_row)), 'constant')
                kp_data = kp_data + data_row
            # divide sums of values over number of rows that qualify
            if df_f.shape[0]:
                kp_data = kp_data / df_f.shape[0]
            # smoothen signal
            if self.smoothen_signal:
                kp_data = self.smoothen_filter(kp_data)
            extracted_data.append({'value': value, 'data': kp_data})
        # build filename
        if not name_file:
            name_file = 'kp_' + variable + '-' + '-'.join(str(val) for val in values)
        # plotly figure
        fig = subplots.make_subplots(rows=1, cols=1, shared_xaxes=False)
        # plot each variable in data
        for data in range(len(extracted_data)):
            # custom labels for legend
            if y_legend:
                name = y_legend[data]
            else:
                name = str(extracted_data[data]['value'])
            fig.add_trace(go.Scatter(y=extracted_data[data]['data'],
                                     mode='lines',
                                     x=times,
                                     name=name),
                          row=1,
                          col=1)
        # update layout
        fig.update_layout(template=self.template,
                          xaxis_title=xaxis_title,
                          yaxis_title=yaxis_title,
                          xaxis_range=xaxis_range,
                          yaxis_range=yaxis_range)
        # draw events
        self.draw_events(fig=fig,
                         yaxis_range=yaxis_range,
                         events=events,
                         events_width=events_width,
                         events_dash=events_dash,
                         events_colour=events_colour,
                         events_annotations_font_size=events_annotations_font_size,
                         events_annotations_colour=events_annotations_colour)
        # draw ttest and anova rows
        # self.draw_ttest_anova(fig=fig,
        #                       times=times,
        #                       name_file=name_file,
        #                       yaxis_range=yaxis_range,
        #                       yaxis_step=yaxis_step,
        #                       ttest_signals=ttest_signals,
        #                       ttest_marker=ttest_marker,
        #                       ttest_marker_size=ttest_marker_size,
        #                       ttest_marker_colour=ttest_marker_colour,
        #                       ttest_annotations_font_size=ttest_annotations_font_size,
        #                       ttest_annotations_colour=ttest_annotations_colour,
        #                       anova_signals=anova_signals,
        #                       anova_marker=anova_marker,
        #                       anova_marker_size=anova_marker_size,
        #                       anova_marker_colour=anova_marker_colour,
        #                       anova_annotations_font_size=anova_annotations_font_size,
        #                       anova_annotations_colour=anova_annotations_colour,
        #                       ttest_anova_row_height=ttest_anova_row_height)
        # create tabs
        buttons = list([dict(label='All',
                             method='update',
                             args=[{'visible': [True] * len(values)}, {'title': 'All', 'showlegend': True}])])
        # show menu with selection of variable to plot
        if show_menu:
            # counter for traversing through stimuli
            counter_rows = 0
            for value in values:
                visibility = [[counter_rows == j] for j in range(len(values))]
                visibility = [item for sublist in visibility for item in sublist]
                button = dict(label=str(value),
                              method='update',
                              args=[{'visible': visibility}, {'title': str(value)}])
                buttons.append(button)
                counter_rows = counter_rows + 1
            # add menu
            updatemenus = [dict(x=-0.15, buttons=buttons, showactive=True)]
            fig['layout']['updatemenus'] = updatemenus
        # update layout
        if show_title:
            fig['layout']['title'] = 'Keypresses for ' + variable
        # update font family
        if font_family:
            # use given value
            fig.update_layout(font=dict(family=font_family))
        else:
            # use value from config file
            fig.update_layout(font=dict(family=tr.common.get_configs('font_family')))
        # update font size
        if font_size:
            # use given value
            fig.update_layout(font=dict(size=font_size))
        else:
            # use value from config file
            fig.update_layout(font=dict(size=tr.common.get_configs('font_size')))
        # legend
        if legend_x and legend_y:
            fig.update_layout(legend=dict(x=legend_x, y=legend_y))
        # save file to local output folder
        if save_file:
            self.save_plotly(fig=fig,
                             name=name_file,
                             remove_margins=True,
                             width=fig_save_width,
                             height=fig_save_height,
                             save_final=save_final)  # also save as "final" figure
        # open it in localhost instead
        else:
            fig.show()

    # def conduct_mixed_anova_questions(self, batch_dir, output_dir=None):
    #     """
    #     Perform 2-way mixed ANOVA for post-stimulus questions for each batch file.

    #     Args:
    #         batch_dir (str): Directory containing batch files (e.g., batch_*.csv).
    #         output_dir (str): Directory to save ANOVA results.

    #     Returns:
    #         None
    #     """
    #     if output_dir is None:
    #         output_dir = self.anova_dir
    #     logger.info(f"Starting 2-way mixed ANOVA for post-stimulus questions in {batch_dir}...")

    #     os.makedirs(output_dir, exist_ok=True)

    #     for batch_file in sorted(os.listdir(batch_dir)):
    #         if not batch_file.startswith('batch_') or not batch_file.endswith('.csv'):
    #             continue

    #         batch_path = os.path.join(batch_dir, batch_file)

    #         if os.path.getsize(batch_path) == 0:
    #             logger.warning(f"Skipping empty batch file: {batch_file}")
    #             continue

    #         try:
    #             batch_data = pd.read_csv(batch_path)
    #         except Exception as e:
    #             logger.error(f"Failed to read batch file {batch_file}: {e}")
    #             continue

    #         logger.info(f"Processing {batch_file} for ANOVA...")

    #         anova_results = []

    #         # Process each question
    #         for question_col in [col for col in batch_data.columns if col.startswith('question_')]:
    #             for time_bin in batch_data['TimeBin'].dropna().unique():
    #                 time_bin_data = batch_data[batch_data['TimeBin'] == time_bin].dropna(subset=[question_col])

    #                 if len(time_bin_data['EgoCar'].unique()) < 2:
    #                     logger.warning(f"Skipping TimeBin {time_bin} for {question_col}: only one level of EgoCar.")
    #                     continue

    #                 if len(time_bin_data['TargetCar'].unique()) < 2:
    #                     logger.warning(f"Skipping TimeBin {time_bin} for {question_col}: only one level of TargetCar.")
    #                     continue

    #                 if time_bin_data[question_col].nunique() <= 1:
    #                     logger.warning(f"Skipping TimeBin {time_bin} for {question_col}: insufficient variability.")
    #                     continue

    #                 try:
    #                     # Prepare data for Pingouin's mixed_anova
    #                     time_bin_data['TargetCar'] = time_bin_data['TargetCar'].astype('category')
    #                     time_bin_data['EgoCar'] = time_bin_data['EgoCar'].astype('category')

    #                     # Run ANOVA
    #                     aov = pg.mixed_anova(
    #                         dv=question_col,
    #                         within='TargetCar',
    #                         between='EgoCar',
    #                         subject='ParticipantID',
    #                         data=time_bin_data
    #                     )

    #                     results = {
    #                         'TimeBin': time_bin,
    #                         'Question': question_col,
    #                         'Within-TargetCar-F': aov.loc[aov['Source'] == 'TargetCar', 'F'].values[0],
    #                         'Within-TargetCar-p': aov.loc[aov['Source'] == 'TargetCar', 'p-unc'].values[0],
    #                         'Interaction-F': aov.loc[aov['Source'] == 'Interaction', 'F'].values[0],
    #                         'Interaction-p': aov.loc[aov['Source'] == 'Interaction', 'p-unc'].values[0],
    #                         'Between-EgoCar-F': aov.loc[aov['Source'] == 'EgoCar', 'F'].values[0],
    #                         'Between-EgoCar-p': aov.loc[aov['Source'] == 'EgoCar', 'p-unc'].values[0]
    #                     }

    #                     anova_results.append(results)

    #                 except Exception as e:
    #                     logger.error(f"ANOVA failed for {question_col}, TimeBin {time_bin}: {e}")

    #         # Save results if any exist
    #         if anova_results:
    #             anova_file = os.path.join(output_dir, f"{batch_file.replace('.csv', '_questions_anova_results.csv')}")
    #             pd.DataFrame(anova_results).to_csv(anova_file, index=False)
    #             logger.info(f"ANOVA results for questions saved to {anova_file}.")
    #         else:
    #             logger.warning(f"No ANOVA results generated for {batch_file}.")

    def plot_kp_variables_or(self, df, variables, y_legend=None, xaxis_title='Time (s)',
                             yaxis_title='Percentage of trials with response key pressed', xaxis_range=None,
                             yaxis_range=None, show_menu=False, show_title=True, name_file=None, save_file=False,
                             save_final=False, fig_save_width=1320, fig_save_height=680, legend_x=0, legend_y=0,
                             font_family=None, font_size=None, events=None, events_width=1, events_dash='dot',
                             events_colour='black', events_annotations_font_size=20, events_annotations_colour='black',
                             ttest_signals=None, ttest_marker='circle', ttest_marker_size=3,
                             ttest_marker_colour='black', ttest_annotations_font_size=10,
                             ttest_annotations_colour='black', anova_signals=None, anova_marker='cross',
                             anova_marker_size=3, anova_marker_colour='black', anova_annotations_font_size=10,
                             anova_annotations_colour='black', ttest_anova_row_height=0.5, yaxis_step=10):
        """Separate plots of keypresses with multiple variables as a filter.

        Args:
            df (dataframe): dataframe with keypress data.
            variables (list): variables to plot.
            y_legend (list, optional): names for variables to be shown in the legend.
            xaxis_title (str, optional): title for x axis.
            yaxis_title (str, optional): title for y axis.
            xaxis_range (list, optional): range of x axis in format [min, max].
            yaxis_range (list, optional): range of y axis in format [min, max].
            show_menu (bool, optional): show menu on top left with variables to select for plotting.
            show_title (bool, optional): show title on top of figure.
            name_file (str, optional): name of file to save.
            save_file (bool, optional): flag for saving an html file with plot.
            save_final (bool, optional): flag for saving an a final figure to /figures.
            fig_save_width (int, optional): width of figures to be saved.
            fig_save_height (int, optional): height of figures to be saved.
            legend_x (float, optional): location of legend, percentage of x axis. 0 = use default value.
            legend_y (float, optional): location of legend, percentage of y axis. 0 = use default value.
            font_family (str, optional): font family to be used across the figure. None = use config value.
            font_size (int, optional): font size to be used across the figure. None = use config value.
            events (list, optional): list of events to draw formatted as values on x axis.
            events_width (int, optional): thickness of the vertical lines.
            events_dash (str, optional): type of the vertical lines.
            events_colour (str, optional): colour of the vertical lines.
            events_annotations_font_size (int, optional): font size of annotations for the vertical lines.
            events_annotations_colour (str, optional): colour of annotations for the vertical lines.
            ttest_signals (list, optional): signals to compare with ttest. None = do not compare.
            ttest_marker (str, optional): symbol of markers for the ttest.
            ttest_marker_size (int, optional): size of markers for the ttest.
            ttest_marker_colour (str, optional): colour of markers for the ttest.
            ttest_annotations_font_size (int, optional): font size of annotations for ttest.
            ttest_annotations_colour (str, optional): colour of annotations for ttest.
            anova_signals (dict, optional): signals to compare with ANOVA. None = do not compare.
            anova_marker (str, optional): symbol of markers for the ANOVA.
            anova_marker_size (int, optional): size of markers for the ANOVA.
            anova_marker_colour (str, optional): colour of markers for the ANOVA.
            anova_annotations_font_size (int, optional): font size of annotations for ANOVA.
            anova_annotations_colour (str, optional): colour of annotations for ANOVA.
            ttest_anova_row_height (int, optional): height of row of ttest/anova markers.
            yaxis_step (int): step between ticks on y axis.
            ttest_anova_row_height (int, optional): height of row of ttest/anova markers.
            yaxis_step (int): step between ticks on y axis.
        """
        logger.info('Creating visualisation of keypresses based on variables {} with OR filter.', variables)
        # build string with variables
        variables_str = ''
        for variable in variables:
            variables_str = variables_str + '_' + str(variable['variable']) + '-' + str(variable['value'])
        # calculate times
        times = np.array(range(self.res, df['video_length'].max() + self.res, self.res)) / 1000
        # extract data for values
        extracted_data = []
        for var in variables:
            kp_data = np.array([0.0] * len(times))
            # non-nan value (provide as np.nan)
            if not pd.isnull(var['value']):
                df_f = df[df[var['variable']] == var['value']]
            # nan value
            else:
                df_f = df[var['variable'].isnull()]
            # go over extracted videos
            for index, row in df_f.iterrows():
                kp_data = kp_data + np.array(row['kp'])
            # divide sums of values over number of rows that qualify
            if df_f.shape[0]:
                kp_data = kp_data / df_f.shape[0]
            # smoothen signal
            if self.smoothen_signal:
                kp_data = self.smoothen_filter(kp_data)
            extracted_data.append({'value': str(var['variable']) + '-' + str(var['value']), 'data': kp_data})
        # build filename
        if not name_file:
            name_file = 'kp_or' + variables_str
        # plotly figure
        fig = subplots.make_subplots(rows=1, cols=1, shared_xaxes=True)
        # plot each variable in data
        for data in range(len(extracted_data)):
            # custom labels for legend
            if y_legend:
                name = y_legend[data]
            else:
                name = str(extracted_data[data]['value'])
            fig.add_trace(go.Scatter(y=extracted_data[data]['data'],
                                     mode='lines',
                                     x=times,
                                     name=name),
                          row=1,
                          col=1)
        # update layout
        fig.update_layout(template=self.template,
                          xaxis_title=xaxis_title,
                          yaxis_title=yaxis_title,
                          xaxis_range=xaxis_range,
                          yaxis_range=yaxis_range)
        # draw events
        self.draw_events(fig=fig,
                         yaxis_range=yaxis_range,
                         events=events,
                         events_width=events_width,
                         events_dash=events_dash,
                         events_colour=events_colour,
                         events_annotations_font_size=events_annotations_font_size,
                         events_annotations_colour=events_annotations_colour)
        # draw ttest and anova rows
        self.draw_ttest_anova(fig=fig,
                              times=times,
                              name_file=name_file,
                              yaxis_range=yaxis_range,
                              yaxis_step=yaxis_step,
                              ttest_signals=ttest_signals,
                              ttest_marker=ttest_marker,
                              ttest_marker_size=ttest_marker_size,
                              ttest_marker_colour=ttest_marker_colour,
                              ttest_annotations_font_size=ttest_annotations_font_size,
                              ttest_annotations_colour=ttest_annotations_colour,
                              anova_signals=anova_signals,
                              anova_marker=anova_marker,
                              anova_marker_size=anova_marker_size,
                              anova_marker_colour=anova_marker_colour,
                              anova_annotations_font_size=anova_annotations_font_size,
                              anova_annotations_colour=anova_annotations_colour,
                              ttest_anova_row_height=ttest_anova_row_height)
        # create tabs
        buttons = list([dict(label='All',
                             method='update',
                             args=[{'visible': [True] * len(variables)}, {'title': 'All', 'showlegend': True}])])
        # show menu with selection of variable to plot
        if show_menu:
            # create tabs
            buttons = list([dict(label='All',
                                 method='update',
                                 args=[{'visible': [True] * len(variables)},
                                       {'title': 'All',
                                        'showlegend': True}])])
            # counter for traversing through stimuli
            counter_rows = 0
            for data in extracted_data:
                visibility = [[counter_rows == j] for j in range(len(variables))]
                visibility = [item for sublist in visibility for item in sublist]
                button = dict(label=data['value'],
                              method='update',
                              args=[{'visible': visibility},
                                    {'title': data['value']}])
                buttons.append(button)
                counter_rows = counter_rows + 1
            # add menu
            updatemenus = [dict(x=-0.15, buttons=buttons, showactive=True)]
            fig['layout']['updatemenus'] = updatemenus
        # update layout
        if show_title:
            fig['layout']['title'] = 'Keypresses for ' + variable
        # update font family
        if font_family:
            # use given value
            fig.update_layout(font=dict(family=font_family))
        else:
            # use value from config file
            fig.update_layout(font=dict(family=tr.common.get_configs('font_family')))
        # update font size
        if font_size:
            # use given value
            fig.update_layout(font=dict(size=font_size))
        else:
            # use value from config file
            fig.update_layout(font=dict(size=tr.common.get_configs('font_size')))
        # legend
        if legend_x and legend_y:
            fig.update_layout(legend=dict(x=legend_x, y=legend_y))
        # save file to local output folder
        if save_file:
            self.save_plotly(fig=fig,
                             name=name_file,
                             remove_margins=True,
                             width=fig_save_width,
                             height=fig_save_height,
                             save_final=save_final)  # also save as "final" figure
        # open it in localhost instead
        else:
            fig.show()

    def plot_kp_variables_and(self, df, plot_names, variables, conf_interval=None, xaxis_title='Time (s)',
                              yaxis_title='Percentage of trials with response key pressed',
                              xaxis_range=None, yaxis_range=None, name_file=None, save_file=False,
                              save_final=False, fig_save_width=1320, fig_save_height=680, font_family=None,
                              font_size=None, events=None, events_width=1, events_dash='dot',
                              events_colour='black', events_annotations_font_size=20,
                              events_annotations_colour='black', ttest_signals=None, ttest_marker='circle',
                              ttest_marker_size=3, ttest_marker_colour='black', ttest_annotations_font_size=10,
                              ttest_annotations_colour='black', anova_signals=None, anova_marker='cross',
                              anova_marker_size=3, anova_marker_colour='black', anova_annotations_font_size=10,
                              anova_annotations_colour='black', ttest_anova_row_height=0.5, yaxis_step=10):
        """Separate plots of keypresses with multiple variables as a filter.

        Args:
            df (dataframe): dataframe with keypress data.
            plot_names (list): names of plots.
            variables (list): variables to plot.
            conf_interval (float, optional): show confidence interval defined by argument.
            xaxis_title (str, optional): title for x axis.
            yaxis_title (str, optional): title for y axis.
            xaxis_range (list, optional): range of x axis in format [min, max].
            yaxis_range (list, optional): range of y axis in format [min, max].
            name_file (str, optional): name of file to save.
            save_file (bool, optional): flag for saving an html file with plot.
            save_final (bool, optional): flag for saving an a final figure to /figures.
            fig_save_width (int, optional): width of figures to be saved.
            fig_save_height (int, optional): height of figures to be saved.
            font_family (str, optional): font family to be used across the figure. None = use config value.
            font_size (int, optional): font size to be used across the figure. None = use config value.
            events (list, optional): list of events to draw formatted as values on x axis.
            events_width (int, optional): thickness of the vertical lines.
            events_dash (str, optional): type of the vertical lines.
            events_colour (str, optional): colour of the vertical lines.
            events_annotations_font_size (int, optional): font size of annotations for the vertical lines.
            events_annotations_colour (str, optional): colour of annotations for the vertical lines.
            ttest_signals (list, optional): signals to compare with ttest. None = do not compare.
            ttest_marker (str, optional): symbol of markers for the ttest.
            ttest_marker_size (int, optional): size of markers for the ttest.
            ttest_marker_colour (str, optional): colour of markers for the ttest.
            ttest_annotations_font_size (int, optional): font size of annotations for ttest.
            ttest_annotations_colour (str, optional): colour of annotations for ttest.
            anova_signals (dict, optional): signals to compare with ANOVA. None = do not compare.
            anova_marker (str, optional): symbol of markers for the ANOVA.
            anova_marker_size (int, optional): size of markers for the ANOVA.
            anova_marker_colour (str, optional): colour of markers for the ANOVA.
            anova_annotations_font_size (int, optional): font size of annotations for ANOVA.
            anova_annotations_colour (str, optional): colour of annotations for ANOVA.
            ttest_anova_row_height (int, optional): height of row of ttest/anova markers.
            yaxis_step (int): step between ticks on y axis.
            ttest_anova_row_height (int, optional): height of row of ttest/anova markers.
            yaxis_step (int): step between ticks on y axis.
        """
        logger.info('Creating visualisation of keypresses based on variables {} with AND filter.', variables)
        # build string with variables
        # create an empty figure, to add scatters to
        fig = subplots.make_subplots(rows=1, cols=1, shared_xaxes=True)
        counter = 0
        # retrieve lists to make combined AND plot
        for variables in variables:
            variables_str = ''
            for variable in variables:
                variables_str = variables_str + '_' + str(variable['variable'])
            # calculate times
            times = np.array(range(self.res, df['video_length'].max() + self.res, self.res)) / 1000
            # filter df based on variables given
            for var in variables:
                # non-nan value (provide as np.nan)
                if not pd.isnull(var['value']):
                    df_f = df[df[var['variable']] == var['value']]
                # nan value
                else:
                    df_f = df[df[var['variable']].isnull()]
            # check if any data in df left
            if df_f.empty:
                logger.error('Provided variables yielded empty dataframe.')
                return
            # add all data together. Must be converted to np array
            kp_data = np.array([0.0] * len(times))
            # go over extracted videos
            for i, data in enumerate(df_f['kp']):
                kp_data += np.array(data)
            # divide sums of values over number of rows that qualify
            if df_f.shape[0]:
                kp_data = kp_data / df_f.shape[0]
            # smoothen signal
            if self.smoothen_signal:
                kp_data = self.smoothen_filter(kp_data)
            # plot each variable in data
            fig.add_trace(go.Scatter(y=kp_data,
                                     mode='lines',
                                     x=times,
                                     name=plot_names[counter]),
                          row=1,
                          col=1)
            # show confidence interval
            if conf_interval:
                # calculate confidence interval
                (y_lower, y_upper) = self.get_conf_interval_bounds(kp_data, conf_interval)
                # plot interval
                fig.add_trace(go.Scatter(name='Upper bound',
                                         x=times,
                                         y=y_upper,
                                         mode='lines',
                                         fillcolor='rgba(0,100,80,0.2)',
                                         line=dict(color='rgba(255,255,255,0)'),
                                         hoverinfo="skip",
                                         showlegend=False))
                fig.add_trace(go.Scatter(name='Lower bound',
                                         x=times,
                                         y=y_lower,
                                         fill='tonexty',
                                         fillcolor='rgba(0,100,80,0.2)',
                                         line=dict(color='rgba(255,255,255,0)'),
                                         hoverinfo="skip",
                                         showlegend=False))
            # define range of y axis
            if not yaxis_range:
                yaxis_range = [0, max(y_upper) if conf_interval else max(kp_data)]
            # update layout
            fig.update_layout(template=self.template,
                              xaxis_title=xaxis_title,
                              yaxis_title=yaxis_title,
                              xaxis_range=xaxis_range,
                              yaxis_range=yaxis_range)
            counter = counter + 1
        # draw events
        self.draw_events(fig=fig,
                         yaxis_range=yaxis_range,
                         events=events,
                         events_width=events_width,
                         events_dash=events_dash,
                         events_colour=events_colour,
                         events_annotations_font_size=events_annotations_font_size,
                         events_annotations_colour=events_annotations_colour)
        # draw ttest and anova rows
        self.draw_ttest_anova(fig=fig,
                              times=times,
                              name_file=name_file,
                              yaxis_range=yaxis_range,
                              yaxis_step=yaxis_step,
                              ttest_signals=ttest_signals,
                              ttest_marker=ttest_marker,
                              ttest_marker_size=ttest_marker_size,
                              ttest_marker_colour=ttest_marker_colour,
                              ttest_annotations_font_size=ttest_annotations_font_size,
                              ttest_annotations_colour=ttest_annotations_colour,
                              anova_signals=anova_signals,
                              anova_marker=anova_marker,
                              anova_marker_size=anova_marker_size,
                              anova_marker_colour=anova_marker_colour,
                              anova_annotations_font_size=anova_annotations_font_size,
                              anova_annotations_colour=anova_annotations_colour,
                              ttest_anova_row_height=ttest_anova_row_height)
        # update font family
        if font_family:
            # use given value
            fig.update_layout(font=dict(family=font_family))
        else:
            # use value from config file
            fig.update_layout(font=dict(family=tr.common.get_configs('font_family')))
        # update font size
        if font_size:
            # use given value
            fig.update_layout(font=dict(size=font_size))
        else:
            # use value from config file
            fig.update_layout(font=dict(size=tr.common.get_configs('font_size')))
        # save file to local output folder
        if save_file:
            # build filename
            if not name_file:
                name_file = 'kp_and' + variables_str
            self.save_plotly(fig=fig,
                             name=name_file,
                             remove_margins=True,
                             width=fig_save_width,
                             height=fig_save_height,
                             save_final=save_final)  # also save as "final" figure
        # open it in localhost instead
        else:
            fig.show()

    def map(self, df, color, name_file=None, save_file=False, save_final=False, fig_save_width=1320,
            fig_save_height=680, font_family=None, font_size=None):
        """Map of countries of participation with colour based on column in dataframe.

        Args:
            df (dataframe): dataframe with keypress data.
            name_file (str, optional): name of file to save.
            save_file (bool, optional): flag for saving an html file with plot.
            save_final (bool, optional): flag for saving an a final figure to /figures.
            fig_save_width (int, optional): width of figures to be saved.
            fig_save_height (int, optional): height of figures to be saved.
            font_family (str, optional): font family to be used across the figure. None = use config value.
            font_size (int, optional): font size to be used across the figure. None = use config value.
        """
        logger.info('Creating visualisation of heatmap of participants by  country with colour defined by {}.', color)
        # create map
        fig = px.choropleth(df,
                            locations='country',
                            color=color,
                            hover_name='country',
                            color_continuous_scale=px.colors.sequential.Plasma)
        # update layout
        fig.update_layout(template=self.template)
        # update font family
        if font_family:
            # use given value
            fig.update_layout(font=dict(family=font_family))
        else:
            # use value from config file
            fig.update_layout(font=dict(family=tr.common.get_configs('font_family')))
        # update font size
        if font_size:
            # use given value
            fig.update_layout(font=dict(size=font_size))
        else:
            # use value from config file
            fig.update_layout(font=dict(size=tr.common.get_configs('font_size')))
        # save file to local output folder
        if save_file:
            # build filename
            if not name_file:
                name_file = 'map_' + color
            self.save_plotly(fig=fig,
                             name=name_file,
                             remove_margins=True,
                             width=fig_save_width,
                             height=fig_save_height,
                             save_final=save_final)  # also save as "final" figure
        # open it in localhost instead
        else:
            fig.show()

    def save_plotly(self, fig, name, remove_margins=False, width=1320, height=680, save_eps=True, save_png=True,
                    save_html=True, open_browser=True, save_mp4=False, save_final=False):
        """
        Helper function to save figure as html file.

        Args:
            fig (plotly figure): figure object.
            name (str): name of html file.
            path (str): folder for saving file.
            remove_margins (bool, optional): remove white margins around EPS figure.
            width (int, optional): width of figures to be saved.
            height (int, optional): height of figures to be saved.
            save_eps (bool, optional): save image as EPS file.
            save_png (bool, optional): save image as PNG file.
            save_html (bool, optional): save image as html file.
            open_browser (bool, optional): open figure in the browse.
            save_mp4 (bool, optional): save video as MP4 file.
            save_final (bool, optional): whether to save the "good" final figure.
        """
        # build path
        path = os.path.join(tr.settings.output_dir, self.folder_figures)
        if not os.path.exists(path):
            os.makedirs(path)
        # build path for final figure
        path_final = os.path.join(tr.settings.root_dir, self.folder_figures)
        if save_final and not os.path.exists(path_final):
            os.makedirs(path_final)
        # limit name to max 200 char (for Windows)
        if len(path) + len(name) > 195 or len(path_final) + len(name) > 195:
            name = name[:200 - len(path) - 5]
        # save as html
        if save_html:
            if open_browser:
                # open in browser
                py.offline.plot(fig, filename=os.path.join(path, name + '.html'))
                # also save the final figure
                if save_final:
                    py.offline.plot(fig, filename=os.path.join(path_final, name + '.html'), auto_open=False)
            else:
                # do not open in browser
                py.offline.plot(fig, filename=os.path.join(path, name + '.html'), auto_open=False)
                # also save the final figure
                if save_final:
                    py.offline.plot(fig, filename=os.path.join(path_final, name + '.html'), auto_open=False)
        # remove white margins
        if remove_margins:
            fig.update_layout(margin=dict(l=2, r=2, t=20, b=12))
        # save as eps
        if save_eps:
            fig.write_image(os.path.join(path, name + '.eps'), width=width, height=height)
            # also save the final figure
            if save_final:
                fig.write_image(os.path.join(path_final, name + '.eps'), width=width, height=height)
        # save as png
        if save_png:
            fig.write_image(os.path.join(path, name + '.png'), width=width, height=height)
            # also save the final figure
            if save_final:
                fig.write_image(os.path.join(path_final, name + '.png'), width=width, height=height)
        # save as mp4
        if save_mp4:
            fig.write_image(os.path.join(path, name + '.mp4'), width=width, height=height)

    def save_fig(self, fig, name, remove_margins=False, pad_inches=0, save_final=False):
        """
        Helper function to save figure as file.

        Args:
            fig (matplotlib figure): figure object.
            name (str): name of figure to save.
            remove_margins (bool, optional): remove white margins around EPS figure.
            pad_inches (int, optional): padding.
            save_final (bool, optional): whether to save the "good" final figure.
        """
        # build path
        path = os.path.join(tr.settings.output_dir, self.folder_figures)
        if not os.path.exists(path):
            os.makedirs(path)
        # build path for final figure
        path_final = os.path.join(tr.settings.output_dir, self.folder_figures)
        if save_final and not os.path.exists(path_final):
            os.makedirs(path_final)
        # limit name to max 200 char (for Windows)
        if len(path) + len(name) > 195 or len(path_final) + len(name) > 195:
            name = name[:200 - len(path) - 5]
        # remove white margins
        if remove_margins:
            fig.update_layout(margin=dict(l=2, r=2, t=20, b=12))
        # save file
        plt.savefig(os.path.join(path, name), bbox_inches='tight', pad_inches=pad_inches)
        # also save the final figure
        if save_final:
            plt.savefig(os.path.join(path_final, name), bbox_inches='tight', pad_inches=pad_inches)
        # clear figure from memory
        plt.close(fig)

    def save_anim(self, image, anim, name):
        """
        Helper function to save figure as file.

    Args:
            image (image): image to save.
            anim (animatino): animation object.
            name (str): suffix to add to the name of the saved file.
        """
        # build path
        path = os.path.join(tr.settings.output_dir, self.folder_figures)
        if not os.path.exists(path):
            os.makedirs(path)
        # limit name to max 200 char (for Windows)
        if len(path) + len(name) > 195:
            name = name[:200 - len(path) - 5]
        # save file
        anim.save(os.path.join(path, name), writer='ffmpeg')
        # clear animation from memory
        plt.close(self.fig)

    def autolabel(self, ax, on_top=False, decimal=True):
        """
        Attach a text label above each bar in, displaying its height.

        Args:
            ax (matplotlib axis): bas objects in figure.
            on_top (bool, optional): add labels on top of bars.
            decimal (bool, optional): add 2 decimal digits.
        """
        # todo: optimise to use the same method
        # on top of bar
        if on_top:
            for rect in ax.patches:
                height = rect.get_height()
                # show demical points
                if decimal:
                    label_text = f'{height:.2f}'
                else:
                    label_text = f'{height:.0f}'
                ax.annotate(label_text,
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center',
                            va='bottom')
        # in the middle of the bar
        else:
            # based on https://stackoverflow.com/a/60895640/46687
            # .patches is everything inside of the chart
            for rect in ax.patches:
                # Find where everything is located
                height = rect.get_height()
                width = rect.get_width()
                x = rect.get_x()
                y = rect.get_y()
                # the height of the bar is the data value and can be used as the label show decimal points
                if decimal:
                    label_text = f'{height:.2f}'
                else:
                    label_text = f'{height:.0f}'
                label_x = x + width / 2
                label_y = y + height / 2
                # plot only when height is greater than specified value
                if height > 0:
                    ax.text(label_x, label_y, label_text, ha='center', va='center')

    def reset_font(self):
        """
        Reset font to default size values. Info at
        https://matplotlib.org/tutorials/introductory/customizing.html
        """
        s_font = 8
        m_font = 10
        l_font = 12
        plt.rc('font', size=s_font)         # controls default text sizes
        plt.rc('axes', titlesize=s_font)    # fontsize of the axes title
        plt.rc('axes', labelsize=m_font)    # fontsize of the axes labels
        plt.rc('xtick', labelsize=s_font)   # fontsize of the tick labels
        plt.rc('ytick', labelsize=s_font)   # fontsize of the tick labels
        plt.rc('legend', fontsize=s_font)   # legend fontsize
        plt.rc('figure', titlesize=l_font)  # fontsize of the figure title

    def get_conf_interval_bounds(self, data, conf_interval=0.95):
        """Get lower and upper bounds of confidence interval.

        Args:
            data (list): list with data.
            conf_interval (float, optional): confidence interval value.

        Returns:
            list of lists: lower and upper bounds.
        """
        # calculate confidence interval
        conf_interval = st.t.interval(conf_interval, len(data) - 1, loc=np.mean(data), scale=st.sem(data))
        # calculate bounds
        y_lower = data - conf_interval[0]
        y_upper = data + conf_interval[1]
        return y_lower, y_upper

    def slugify(self, value, allow_unicode=False):
        """
        Taken from https://github.com/django/django/blob/master/django/utils/text.py
        Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated dashes to single dashes. Remove
        characters that aren't alphanumerics, underscores, or hyphens. Convert to lowercase. Also strip leading and
        trailing white space, dashes, and underscores.
        """
        value = str(value)
        if allow_unicode:
            value = unicodedata.normalize('NFKC', value)
        else:
            value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
        value = re.sub(r'[^\w\s-]', '', value.lower())
        return re.sub(r'[-\s]+', '-', value).strip('-_')

    def smoothen_filter(self, signal, type_flter='OneEuroFilter'):
        """Smoothen list with a filter.

        Args:
            signal (list): input signal to smoothen
            type_flter (str, optional): type_flter of filter to use.

        Returns:
            list: list with smoothened data.
        """
        if type_flter == 'OneEuroFilter':
            filter_kp = OneEuroFilter(freq=tr.common.get_configs('freq'),            # frequency
                                      mincutoff=tr.common.get_configs('mincutoff'),  # minimum cutoff frequency
                                      beta=tr.common.get_configs('beta'))            # beta value
            return [filter_kp(value) for value in signal]
        else:
            logger.error('Specified filter {} not implemented.', type_flter)
            return -1

    def ttest(self, signal_1, signal_2, type='two-sided', paired=True):
        """
        Perform a t-test on two signals, computing p-values and significance.

        Args:
            signal_1 (list): First signal, a list of numeric values.
            signal_2 (list): Second signal, a list of numeric values.
            type (str, optional): Type of t-test to perform. Options are "two-sided",
                                  "greater", or "less". Defaults to "two-sided".
            paired (bool, optional): Indicates whether to perform a paired t-test
                                     (`ttest_rel`) or an independent t-test (`ttest_ind`).
                                     Defaults to True (paired).

        Returns:
            list: A list containing two elements:
                  - p_values (list): Raw p-values for each bin.
                  - significance (list): Binary flags (0 or 1) indicating whether
                    the p-value for each bin is below the threshold configured in
                    `tr.common.get_configs('p_value')`.
        """
        # Check if the lengths of the two signals are the same
        if len(signal_1) != len(signal_2):
            logger.error('The lengths of signal_1 and signal_2 must be the same.')
            return -1
        # convert to numpy arrays if signal_1 and signal_2 are lists
        signal_1 = np.asarray(signal_1)
        signal_2 = np.asarray(signal_2)
        p_values = []  # record raw p value for each bin
        significance = []  # record binary flag (0 or 1) if p value < tr.common.get_configs('p_value'))
        # perform t-test for each value (treated as an independent bin)
        for i in range(len(signal_1)):
            if paired:
                t_stat, p_value = ttest_rel([signal_1[i]], [signal_2[i]], axis=-1, alternative=type)
            else:
                t_stat, p_value = ttest_ind([signal_1[i]], [signal_2[i]], axis=-1, alternative=type, equal_var=False)
            # record raw p value
            p_values.append(p_value)
            # determine significance for this value
            significance.append(int(p_value < tr.common.get_configs('p_value')))
        # return raw p values and binary flags for significance for output
        return [p_values, significance]



    def conduct_rank_transformed_anova_for_questions(self, output_dir, questions=None):
        """
        Perform rank-transformed 2-way mixed ANOVA and homogeneity tests for post-stimulus questions.

        Args:
            questions (list): List of questions to analyze.

        Returns:
            None
        """
        stats_dir = os.path.join(tr.settings.output_dir, self.folder_stats)  # Use consistent stats directory
        os.makedirs(stats_dir, exist_ok=True)

        logger.info(f"Starting rank-transformed ANOVA with homogeneity tests for questions batch files in {tr.settings.output_dir}...")
        batch_files = [
            f for f in sorted(os.listdir(tr.settings.output_dir))
            if f.startswith('batch_') and f.endswith('_question_data.csv')
        ]

        if not batch_files:
            logger.warning(f"No valid batch files found in {tr.settings.output_dir}.")
            return

        for batch_file in batch_files:
            batch_path = os.path.join(tr.settings.output_dir, batch_file)

            if os.path.getsize(batch_path) == 0:
                logger.warning(f"Skipping empty batch file: {batch_file} (File is empty).")
                continue

            try:
                batch_data = pd.read_csv(batch_path)
            except Exception as e:
                logger.error(f"Failed to read batch file {batch_file}: {e}")
                continue

            if 'VideoNumber' not in batch_data.columns:
                logger.error(f"Missing 'VideoNumber' column in {batch_file}. Skipping file.")
                continue

            logger.info(f"Processing {batch_file} for rank-transformed ANOVA and homogeneity tests...")

            homogeneity_results = []  # Store Levene's test results
            anova_results = []  # Store ANOVA results

            for question in questions:
                question_name = question['question']
                if question_name not in batch_data.columns:
                    logger.warning(f"Question '{question_name}' not found in {batch_file}. Skipping question.")
                    continue

                # Rank transform data for the current question
                batch_data[f'{question_name}_Rank'] = rankdata(batch_data[question_name], method='average')

                # Perform homogeneity test
                try:
                    group_values = [
                        group[f'{question_name}_Rank'].values
                        for _, group in batch_data.groupby(['EgoCar', 'TargetCar'])
                        if len(group) > 1
                    ]
                    if len(group_values) > 1:
                        levene_stat, levene_p = levene(*group_values)
                    else:
                        levene_stat, levene_p = None, None

                    # Log and store homogeneity results
                    homogeneity_results.append({
                        'BatchFile': batch_file,
                        'Question': question_name,
                        'Levene-stat': levene_stat,
                        'Levene-p': levene_p,
                    })

                    if levene_p is not None and levene_p < 0.05:
                        logger.warning(f"Homogeneity assumption violated for Question {question_name} in {batch_file}.")

                except Exception as e:
                    logger.error(f"Levene's test failed for Question {question_name} in {batch_file}: {e}")

                # Perform ANOVA on the 4 videos in the batch for the current question
                try:
                    anova_data = batch_data

                    if len(anova_data['EgoCar'].unique()) < 2 or len(anova_data['TargetCar'].unique()) < 2:
                        logger.warning(f"Skipping ANOVA for Question {question_name} in {batch_file}: insufficient factor levels.")
                        continue

                    aov = pg.mixed_anova(
                        dv=f'{question_name}_Rank',  # Dependent variable (rank-transformed data)
                        within='TargetCar',  # Within-subject factor
                        between='EgoCar',  # Between-subject factor
                        subject='ParticipantID',  # Identifier for subjects
                        data=anova_data
                    )

                    # Store ANOVA results
                    anova_results.append({
                        'BatchFile': batch_file,
                        'Question': question_name,
                        'Within-TargetCar-F': aov.loc[aov['Source'] == 'TargetCar', 'F'].values[0],
                        'Within-TargetCar-p': aov.loc[aov['Source'] == 'TargetCar', 'p-unc'].values[0],
                        'Interaction-F': aov.loc[aov['Source'] == 'Interaction', 'F'].values[0],
                        'Interaction-p': aov.loc[aov['Source'] == 'Interaction', 'p-unc'].values[0],
                        'Between-EgoCar-F': aov.loc[aov['Source'] == 'EgoCar', 'F'].values[0],
                        'Between-EgoCar-p': aov.loc[aov['Source'] == 'EgoCar', 'p-unc'].values[0],
                    })

                except Exception as e:
                    logger.error(f"ANOVA failed for Question {question_name} in {batch_file}: {e}")

            # Save homogeneity results
            if homogeneity_results:
                homogeneity_file = os.path.join(stats_dir, f"{batch_file.replace('.csv', '_homogeneity_results.csv')}")
                pd.DataFrame(homogeneity_results).to_csv(homogeneity_file, index=False)
                logger.info(f"Homogeneity results saved to {homogeneity_file}.")

            # Save ANOVA results
            if anova_results:
                anova_file = os.path.join(stats_dir, f"{batch_file.replace('.csv', '_rank_anova_results.csv')}")
                pd.DataFrame(anova_results).to_csv(anova_file, index=False)
                logger.info(f"Rank-transformed ANOVA results saved to {anova_file}.")

            # Log if no results were generated
            if not homogeneity_results:
                logger.warning(f"No homogeneity results generated for {batch_file}.")
            if not anova_results:
                logger.warning(f"No rank-transformed ANOVA results generated for {batch_file}.")

    def conduct_rank_transformed_anova(self, output_dir):

        """
        Perform rank-transformed 2-way mixed ANOVA and homogeneity tests for each time bin in each video batch file using Pingouin.

        Args:
            batch_dir (str): Directory containing batch files (e.g., batch_*.csv).
            output_dir (str): Directory to save ANOVA and homogeneity test results.

        Returns:
            None
        """
        stats_dir = os.path.join(tr.settings.output_dir, self.folder_stats)
        os.makedirs(stats_dir, exist_ok=True)

        logger.info(f"Processing batch files in directory: {tr.settings.output_dir}")
        batch_files = [
            f for f in os.listdir(tr.settings.output_dir)
            if f.startswith('batch_') and f.endswith('_keypress_data.csv')
        ]
        batch_files = [
            f for f in batch_files if int(f.split('_')[1]) < 21
        ]

        if not batch_files:
            logger.warning(f"No valid batch files found in {tr.settings.output_dir}.")
            return

        for batch_file in tqdm(sorted(batch_files), desc="Processing batch files"):
            logger.info(f"Processing batch file: {batch_file}")
            batch_path = os.path.join(tr.settings.output_dir, batch_file)

            if os.path.getsize(batch_path) == 0:
                logger.warning(f"Skipping empty batch file: {batch_file}")
                continue

            try:
                batch_data = pd.read_csv(batch_path)
                if 'TimeIndex' not in batch_data.columns or 'TimeBin' not in batch_data.columns:
                    logger.error(f"'TimeIndex' or 'TimeBin' column is missing in {batch_file}. Skipping...")
                    continue
            except Exception as e:
                logger.error(f"Failed to read batch file {batch_file}: {e}")
                continue

            all_anova_results = []  # Initialize ANOVA results
            homogeneity_results = []  # Initialize homogeneity test results

            for time_bin in batch_data['TimeBin'].dropna().unique():
                try:
                    time_bin_data = batch_data[batch_data['TimeBin'] == time_bin].dropna(subset=['KPNumber'])
                    # Retrieve TimeIndex corresponding to the TimeBin
                    time_index = batch_data[batch_data['TimeBin'] == time_bin]['TimeIndex'].iloc[0]  # Retrieve TimeIndex for this TimeBin

                # Ensure sufficient data for ANOVA
                    if len(time_bin_data['EgoCar'].unique()) < 2 or len(time_bin_data['TargetCar'].unique()) < 2:
                        logger.warning(f"Skipping TimeBin {time_bin} in {batch_file}: insufficient unique levels.")
                        continue
                    if time_bin_data['KPNumber'].nunique() <= 1:
                        logger.warning(f"Skipping TimeBin {time_bin} in {batch_file}: insufficient variability.")
                        continue
                    # Rank-transform the data
                    time_bin_data['RankedKP'] = time_bin_data['KPNumber'].rank()

                    # Homogeneity Test (Levene's Test)
                    group_values = [
                        group['RankedKP'].values
                        for _, group in time_bin_data.groupby(['EgoCar', 'TargetCar'])
                        if len(group) > 1
                    ]
                    if len(group_values) > 1:
                        levene_stat, levene_p = levene(*group_values)
                    else:
                        levene_stat, levene_p = None, None
                    # Log Homogeneity Test results
                    if levene_p is not None and levene_p < 0.05:
                        logger.warning(f"Homogeneity assumption violated for TimeBin {time_bin} in {batch_file}.")

                    homogeneity_results.append({
                        'BatchFile': batch_file,
                        'TimeBin': time_bin,
                        'TimeIndex': time_index,  # Add TimeIndex to Homogeneity Test results
                        'Levene-stat': levene_stat,
                        'Levene-p': levene_p,
                    })

                    # try:
                    #     time_index = batch_data[batch_data['TimeBin'] == time_bin]['TimeIndex'].iloc[0]
                    # except IndexError:
                    #     logger.error(f"No matching TimeIndex found for TimeBin {time_bin} in {batch_file}.")
                    #     continue

                     # Prepare data for Pingouin's mixed_anova
                    time_bin_data['TargetCar'] = time_bin_data['TargetCar'].astype('category')
                    time_bin_data['EgoCar'] = time_bin_data['EgoCar'].astype('category')

                    # Run mixed ANOVA on ranks
                    aov = pg.mixed_anova(
                        # Dependent variable (ranked keypress numbers)
                        dv='RankedKP',
                        # Within-subject factor
                        within='TargetCar',
                        between='EgoCar',  # Between-subject factor
                        subject='ParticipantID',  # Identifier for subjects
                        data=time_bin_data
                    )
                    results = {
                        'TimeBin': time_bin,
                        'TimeIndex': time_index,  # Add TimeIndex to ANOVA results
                        'Within-TargetCar-F': aov.loc[aov['Source'] == 'TargetCar', 'F'].values[0],
                        'Within-TargetCar-p': aov.loc[aov['Source'] == 'TargetCar', 'p-unc'].values[0],
                        'Within-TargetCar-np2': aov.loc[aov['Source'] == 'TargetCar', 'np2'].values[0],  
                        # Add partial eta squared
                        'Interaction-F': aov.loc[aov['Source'] == 'Interaction', 'F'].values[0],
                        'Interaction-p': aov.loc[aov['Source'] == 'Interaction', 'p-unc'].values[0],
                        'Interaction-np2': aov.loc[aov['Source'] == 'Interaction', 'np2'].values[0],  
                        # Add partial eta squared
                        'Between-EgoCar-F': aov.loc[aov['Source'] == 'EgoCar', 'F'].values[0],
                        'Between-EgoCar-p': aov.loc[aov['Source'] == 'EgoCar', 'p-unc'].values[0],
                        'Between-EgoCar-np2': aov.loc[aov['Source'] == 'EgoCar', 'np2'].values[0]  
                        # Add partial eta squared
                    }
                    # Log when p-values are less than 0.05
                    for key in ['Within-TargetCar-p', 'Interaction-p', 'Between-EgoCar-p']:
                        if results[key] < 0.05:
                            logger.warning(f"Significant p-value ({key}={results[key]}) for TimeBin {time_bin} in {batch_file}.")
                    # Conduct post-hoc tests if interaction is significant
                    if results['Interaction-p'] < 0.05:
                        logger.info(f"Significant interaction found for TimeBin {time_bin} in {batch_file}. Performing post-hoc tests...")
                        results = self.perform_posthoc_tests(time_bin_data, results)

                    all_anova_results.append(results)

                except Exception as e:
                    logger.error(f"ANOVA failed for TimeBin {time_bin} in {batch_file}: {e}")

                # Save Homogeneity Test results
            homogeneity_file = os.path.join(stats_dir, f"{batch_file.replace('.csv', '_homogeneity_results.csv')}")
            if homogeneity_results:
                pd.DataFrame(homogeneity_results).to_csv(homogeneity_file, index=False)
                logger.info(f"Homogeneity results saved to {homogeneity_file}.")
            else:
                logger.warning(f"No homogeneity test results generated for {batch_file}.")

            # Save ANOVA results
            anova_file = os.path.join(stats_dir, f"{batch_file.replace('.csv', '_rank_anova_results.csv')}")
            if all_anova_results:
                pd.DataFrame(all_anova_results).to_csv(anova_file, index=False)
                logger.info(f"Rank-transformed ANOVA results saved to {anova_file}.")
            else:
                logger.warning(f"No rank-transformed ANOVA results generated for {batch_file}.")


    def perform_posthoc_tests(self, data, anova_results):
        """
        Conduct post-hoc pairwise comparisons for interaction effects and append the results to the ANOVA file.

        Args:
            data (DataFrame): Data for a specific time bin (must be rank-transformed).
            anova_results (dict): ANOVA results for this TimeBin to which post-hoc results will be added.

        Returns:
            dict: Updated ANOVA results with post-hoc tests included.
        """
        try:
            logger.info("Starting post-hoc tests for interaction effects.")

            # Ensure the data is rank-transformed
            if 'RankedKP' not in data.columns:
                raise ValueError("Input data is not rank-transformed. 'RankedKP' column is missing.")

            posthoc_results = []

            # Perform within-group pairwise tests
            complete_within_data = data.dropna(subset=['TargetCar', 'RankedKP']).copy()
            participant_counts = complete_within_data.groupby(['ParticipantID', 'TargetCar']).size().unstack()
            valid_participants = participant_counts.dropna().index  # Participants with data in all conditions
            filtered_within_data = complete_within_data[complete_within_data['ParticipantID'].isin(valid_participants)]

            if len(valid_participants) > 1:  # Ensure at least two participants for within-group tests
                within_posthoc = pg.pairwise_tests(
                    dv='RankedKP',
                    within='TargetCar',
                    subject='ParticipantID',
                    padjust=None,  # No automatic adjustment here; we'll handle it manually
                    parametric=False,  # Use non-parametric tests for ranked data
                    data=filtered_within_data
                )
                # Manually adjust p-values
                p_unc = within_posthoc['p-unc'].values
                reject, p_adjust, _, _ = multipletests(p_unc, method='holm')
                within_posthoc['p-adjust'] = p_adjust

                within_posthoc['ContrastType'] = 'Within-Group'
                posthoc_results.append(within_posthoc)
            else:
                logger.warning("Not enough valid participants for within-group post-hoc tests.")

            # Perform between-group pairwise tests (no participant filtering needed)
            between_posthoc = pg.pairwise_tests(
                dv='RankedKP',
                between='EgoCar',
                padjust=None,  # No automatic adjustment here; we'll handle it manually
                parametric=False,  # Use non-parametric tests for ranked data
                data=data
            )
            # Manually adjust p-values
            p_unc = between_posthoc['p-unc'].values
            reject, p_adjust, _, _ = multipletests(p_unc, method='holm')
            between_posthoc['p-adjust'] = p_adjust

            between_posthoc['ContrastType'] = 'Between-Group'
            posthoc_results.append(between_posthoc)

            # Combine post-hoc results
            if posthoc_results:
                combined_posthoc = pd.concat(posthoc_results, ignore_index=True)
                combined_posthoc = combined_posthoc.to_dict(orient='records')
                anova_results['PostHocResults'] = combined_posthoc
                logger.info("Post-hoc tests completed and added to ANOVA results.")
            else:
                anova_results['PostHocResults'] = []
                logger.warning("No post-hoc results were generated.")

            return anova_results

        except Exception as e:
            logger.error(f"Error performing post-hoc tests: {e}")
            return anova_results




    def conduct_nonparametric_tests(self, output_dir):
        """
        Conduct non-parametric tests, report mean, SD, sample size, median, mean rank, and direction of group comparison.

        Args:
            output_dir (str): Directory to save test results.

        Returns:
            None
        """
        stats_dir = os.path.join(tr.settings.output_dir, self.folder_stats)
        os.makedirs(stats_dir, exist_ok=True)
        logger.info(f"Processing batch files in directory: {tr.settings.output_dir}")

        batch_files = [
            f for f in os.listdir(tr.settings.output_dir)
            if f.startswith('batch_') and f.endswith('_keypress_data.csv')
        ]

        if not batch_files:
            logger.warning(f"No valid batch files found in {tr.settings.output_dir}.")
            return

        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        for batch_file in sorted(batch_files):
            logger.info(f"Processing batch file: {batch_file}")
            batch_path = os.path.join(tr.settings.output_dir, batch_file)

            try:
                batch_data = pd.read_csv(batch_path)
                if 'TimeIndex' not in batch_data.columns or 'TimeBin' not in batch_data.columns:
                    logger.error(f"'TimeIndex' or 'TimeBin' column is missing in {batch_file}. Skipping...")
                    continue
            except Exception as e:
                logger.error(f"Failed to read batch file {batch_file}: {e}")
                continue

            test_results = []  # Initialize container for results

            for time_bin in batch_data['TimeBin'].dropna().unique():
                try:
                    time_bin_data = batch_data[batch_data['TimeBin'] == time_bin]
                    time_index = time_bin_data['TimeIndex'].iloc[0]

                    # Within-group tests (TargetCar: 1 vs 0) for each EgoCar level
                    for ego in [0, 1]:
                        within_data = time_bin_data[time_bin_data['EgoCar'] == ego]
                        participant_counts = within_data.groupby(['ParticipantID', 'TargetCar']).size().unstack()
                        valid_participants = participant_counts.dropna().index
                        filtered_data = within_data[within_data['ParticipantID'].isin(valid_participants)]


                        if len(valid_participants) > 1:  # Ensure at least two participants
                            group_0 = filtered_data[filtered_data['TargetCar'] == 0]['KPNumber']
                            group_1 = filtered_data[filtered_data['TargetCar'] == 1]['KPNumber']

                            # Calculate descriptive statistics
                            group_0_mean, group_0_sd, group_0_n = group_0.mean(), group_0.std(), len(group_0)
                            group_1_mean, group_1_sd, group_1_n = group_1.mean(), group_1.std(), len(group_1)
                            group_0_median = group_0.median()
                            group_1_median = group_1.median()
                            logger.info(f"Ego={ego}, TimeBin={time_bin}: Group 0 sample size={len(group_0)}, Group 1 sample size={len(group_1)}")


                            if len(group_0) > 0 and len(group_1) > 0:
                                try:
                                    result = pg.wilcoxon(group_1, group_0, alternative='two-sided')
                                    stat = result['W-val'].iloc[0]
                                    p_value = result['p-val'].iloc[0]

                                    # Calculate ranks
                                    combined = pd.concat([group_0, group_1])
                                    ranks = combined.rank()
                                    group_0_mean_rank = ranks[:len(group_0)].mean()
                                    group_1_mean_rank = ranks[len(group_0):].mean()

                                    # Determine direction based on mean rank
                                    # direction = 'Group1 > Group0' if group_1_mean_rank > group_0_mean_rank else 'Group0 > Group1'
                                    direction = 'AV > MDV' if group_1_mean_rank > group_0_mean_rank else 'MDV > AV'

                                    test_results.append({
                                        'TimeBin': time_bin,
                                        'TimeIndex': time_index,
                                        'Comparison': f'Within Ego={ego}: TargetCar=1 vs TargetCar=0',
                                        'Group0_Mean': group_0_mean,
                                        'Group0_SD': group_0_sd,
                                        'Group0_N': group_0_n,
                                        'Group0_Median': group_0_median,
                                        'Group0_MeanRank': group_0_mean_rank,
                                        'Group1_Mean': group_1_mean,
                                        'Group1_SD': group_1_sd,
                                        'Group1_N': group_1_n,
                                        'Group1_Median': group_1_median,
                                        'Group1_MeanRank': group_1_mean_rank,
                                        'Statistic': stat,
                                        'p-value': p_value,
                                        'Direction': direction
                                    })
                                except Exception as e:
                                    logger.error(f"Error in Wilcoxon test for TimeBin {time_bin}, Ego={ego}: {e}")

                    # Between-group tests (EgoCar: 1 vs 0) for each TargetCar level
                    for target in [0, 1]:
                        between_data = time_bin_data[time_bin_data['TargetCar'] == target]
                        group_0 = between_data[between_data['EgoCar'] == 0]['KPNumber']
                        group_1 = between_data[between_data['EgoCar'] == 1]['KPNumber']

                        # Calculate descriptive statistics
                        group_0_mean, group_0_sd, group_0_n = group_0.mean(), group_0.std(), len(group_0)
                        group_1_mean, group_1_sd, group_1_n = group_1.mean(), group_1.std(), len(group_1)
                        group_0_median = group_0.median()
                        group_1_median = group_1.median()
                        logger.info(f"Target={target}, TimeBin={time_bin}: Group 0 sample size={len(group_0)}, Group 1 sample size={len(group_1)}")

                        if len(group_0) > 0 and len(group_1) > 0:
                            try:
                                stat, p_value = mannwhitneyu(group_1, group_0, alternative='two-sided')

                                # Calculate ranks
                                combined = pd.concat([group_0, group_1])
                                ranks = combined.rank()
                                group_0_mean_rank = ranks[:len(group_0)].mean()
                                group_1_mean_rank = ranks[len(group_0):].mean()

                                # Determine direction based on mean rank
                                # direction = 'Group1 > Group0' if group_1_mean_rank > group_0_mean_rank else 'Group0 > Group1'
                                direction = 'AV > MDV' if group_1_mean_rank > group_0_mean_rank else 'MDV > AV'

                                test_results.append({
                                    'TimeBin': time_bin,
                                    'TimeIndex': time_index,
                                    'Comparison': f'Within Target={target}: EgoCar=1 vs EgoCar=0',
                                    'Group0_Mean': group_0_mean,
                                    'Group0_SD': group_0_sd,
                                    'Group0_N': group_0_n,
                                    'Group0_Median': group_0_median,
                                    'Group0_MeanRank': group_0_mean_rank,
                                    'Group1_Mean': group_1_mean,
                                    'Group1_SD': group_1_sd,
                                    'Group1_N': group_1_n,
                                    'Group1_Median': group_1_median,
                                    'Group1_MeanRank': group_1_mean_rank,
                                    'Statistic': stat,
                                    'p-value': p_value,
                                    'Direction': direction
                                })
                            except Exception as e:
                                logger.error(f"Error in Mann-Whitney U test for TimeBin {time_bin}, Target={target}: {e}")
                    # Additional comparisons across EgoCar and TargetCar combinations
                    comparisons = [
                        {'GroupA': (1, 0), 'GroupB': (0, 1), 'Label': 'Ego=1,Target=0 vs Ego=0,Target=1'},
                        {'GroupA': (1, 1), 'GroupB': (0, 0), 'Label': 'Ego=1,Target=1 vs Ego=0,Target=0'},
                    ]

                    for comp in comparisons:
                        group_a = time_bin_data[
                            (time_bin_data['EgoCar'] == comp['GroupA'][0]) &
                            (time_bin_data['TargetCar'] == comp['GroupA'][1])
                        ]['KPNumber']
                        group_b = time_bin_data[
                            (time_bin_data['EgoCar'] == comp['GroupB'][0]) &
                            (time_bin_data['TargetCar'] == comp['GroupB'][1])
                        ]['KPNumber']


                        if len(group_a) > 0 and len(group_b) > 0:
                            try:
                                stat, p_value = mannwhitneyu(group_a, group_b, alternative='two-sided')

                                # Calculate ranks
                                combined = pd.concat([group_a, group_b])
                                ranks = combined.rank()
                                group_a_mean_rank = ranks[:len(group_a)].mean()
                                group_b_mean_rank = ranks[len(group_a):].mean()

                                # Determine direction
                                if comp['Label'] == 'Ego=1,Target=0 vs Ego=0,Target=1':
                                    direction = '(1,0) > (0,1)' if group_a_mean_rank > group_b_mean_rank else '(0,1) > (1,0)'
                                elif comp['Label'] == 'Ego=1,Target=1 vs Ego=0,Target=0':
                                    direction = '(1,1) > (0,0)' if group_a_mean_rank > group_b_mean_rank else '(0,0) > (1,1)'
                                else:
                                    direction = 'Undetermined'

                                test_results.append({
                                    'TimeBin': time_bin,
                                    'TimeIndex': time_index,
                                    'Comparison': comp['Label'],
                                    'Group0_Mean': group_a.mean(),
                                    'Group0_SD': group_a.std(),
                                    'Group0_N': len(group_a),
                                    'Group0_Median': group_a.median(),
                                    'Group0_MeanRank': group_a_mean_rank,
                                    'Group1_Mean': group_b.mean(),
                                    'Group1_SD': group_b.std(),
                                    'Group1_N': len(group_b),
                                    'Group1_Median': group_b.median(),
                                    'Group1_MeanRank': group_b_mean_rank,
                                    'Statistic': stat,
                                    'p-value': p_value,
                                    'Direction': direction
                                })
                            except Exception as e:
                                logger.error(f"Error in Mann-Whitney U test for {comp['Label']} in TimeBin {time_bin}: {e}")


                except Exception as e:
                    logger.error(f"Error processing TimeBin {time_bin} in {batch_file}: {e}")

            # Save results to CSV
            results_file = os.path.join(stats_dir, f"{batch_file.replace('.csv', '_nonparametric_results.csv')}")
            if test_results:
                pd.DataFrame(test_results).to_csv(results_file, index=False)
                logger.info(f"Non-parametric test results saved to {results_file}.")
            else:
                logger.warning(f"No valid comparisons found for {batch_file}.")





    # def conduct_mixed_anova_pingouin(self, batch_dir, output_dir=None):
    #     """
    #     Perform 2-way mixed ANOVA with normality tests and shifted Box-Cox transformation for every time bin.

    #     Args:
    #         batch_dir (str): Directory containing batch files.
    #         output_dir (str): Directory to save ANOVA results.

    #     Returns:
    #         None
    #     """
    #     if output_dir is None:
    #         output_dir = self.anova_dir  # Default to Heroku's ANOVA directory
    #     logger.info(f"Starting 2-way mixed ANOVA with Box-Cox transformation for batch files in {batch_dir}...")

    #     os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists

    #     for batch_file in sorted(os.listdir(batch_dir)):
    #         if not batch_file.startswith('batch_') or not batch_file.endswith('.csv'):
    #             continue

    #         batch_path = os.path.join(batch_dir, batch_file)

    #         if os.path.getsize(batch_path) == 0:
    #             logger.warning(f"Skipping empty batch file: {batch_file}")
    #             continue

    #         try:
    #             batch_data = pd.read_csv(batch_path)
    #         except Exception as e:
    #             logger.error(f"Failed to read batch file {batch_file}: {e}")
    #             continue

    #         logger.info(f"Processing {batch_file} for ANOVA with Box-Cox transformation...")

    #         anova_results = []  # Initialize results for this batch

    #         for time_bin in batch_data['TimeBin'].dropna().unique():
    #             time_bin_data = batch_data[batch_data['TimeBin'] == time_bin].dropna(subset=['KPNumber'])

    #             if len(time_bin_data['KPNumber'].dropna()) == 0:
    #                 logger.warning(f"No valid data for TimeBin {time_bin} in {batch_file}. Skipping...")
    #                 continue

    #             # Shift KPNumber values for Box-Cox transformation
    #             try:
    #                 shifted_kp = time_bin_data['KPNumber'] + 1  # Add 1 to make all values strictly positive
    #                 transformed_kp, lambda_ = boxcox(shifted_kp)
    #                 time_bin_data['TransformedKP'] = transformed_kp
    #                 logger.info(f"Box-Cox transformation applied (lambda={lambda_:.4f}) for TimeBin {time_bin}.")
    #             except Exception as e:
    #                 logger.warning(f"Box-Cox transformation failed for TimeBin {time_bin}: {e}")
    #                 continue

    #             # Run normality tests
    #             for video_num in time_bin_data['VideoNumber'].unique():
    #                 video_data = time_bin_data[time_bin_data['VideoNumber'] == video_num]
    #                 try:
    #                     if len(video_data['TransformedKP'].dropna()) > 3:
    #                         w_stat, p_value = shapiro(video_data['TransformedKP'])
    #                         results[f'Video-{video_num}-Shapiro-W'] = w_stat
    #                         results[f'Video-{video_num}-Shapiro-p'] = p_value
    #                     else:
    #                         results[f'Video-{video_num}-Shapiro-W'] = None
    #                         results[f'Video-{video_num}-Shapiro-p'] = None
    #                 except Exception as e:
    #                     logger.error(f"Shapiro-Wilk test failed for Video {video_num}, TimeBin {time_bin}: {e}")

    #             # Perform ANOVA on transformed data
    #             try:
    #                 time_bin_data['TargetCar'] = time_bin_data['TargetCar'].astype('category')
    #                 time_bin_data['EgoCar'] = time_bin_data['EgoCar'].astype('category')

    #                 aov = pg.mixed_anova(
    #                     dv='TransformedKP',  # Use transformed KPNumber
    #                     within='TargetCar',
    #                     between='EgoCar',
    #                     subject='ParticipantID',
    #                     data=time_bin_data
    #                 )

    #                 results = {
    #                     'TimeBin': time_bin,
    #                     'Within-TargetCar-F': aov.loc[aov['Source'] == 'TargetCar', 'F'].values[0],
    #                     'Within-TargetCar-p': aov.loc[aov['Source'] == 'TargetCar', 'p-unc'].values[0],
    #                     'Interaction-F': aov.loc[aov['Source'] == 'Interaction', 'F'].values[0],
    #                     'Interaction-p': aov.loc[aov['Source'] == 'Interaction', 'p-unc'].values[0],
    #                     'Between-EgoCar-F': aov.loc[aov['Source'] == 'EgoCar', 'F'].values[0],
    #                     'Between-EgoCar-p': aov.loc[aov['Source'] == 'EgoCar', 'p-unc'].values[0]
    #                 }
    #                 anova_results.append(results)
    #             except Exception as e:
    #                 logger.error(f"ANOVA failed for TimeBin {time_bin} in {batch_file}: {e}")

    #         # Save results if any exist
    #         if anova_results:
    #             anova_file = os.path.join(output_dir, f"{batch_file.replace('.csv', '_anova_results.csv')}")
    #             pd.DataFrame(anova_results).to_csv(anova_file, index=False)
    #             logger.info(f"ANOVA results saved to {anova_file}.")
    #         else:
    #             logger.warning(f"No ANOVA results generated for {batch_file}.")

    def anova(self, signals):
        """
        Perform an ANOVA test on three signals, computing p-values and significance.

        Args:
            signal_1 (list): First signal, a list of numeric values.
            signal_2 (list): Second signal, a list of numeric values.
            signal_3 (list): Third signal, a list of numeric values.

        Returns:
            list: A list containing two elements:
                  - p_values (list): Raw p-values for each bin.
                  - significance (list): Binary flags (0 or 1) indicating whether
                    the p-value for each bin is below the threshold configured in
                    `tr.common.get_configs('p_value')`.
        """
        # check if the lengths of the three signals are the same
        # convert signals to numpy arrays if they are lists
        p_values = []  # record raw p-values for each bin
        significance = []  # record binary flags (0 or 1) if p-value < tr.common.get_configs('p_value')
        # perform ANOVA test for each value (treated as an independent bin)
        transposed_data = list(zip(*signals['signals']))
        for i in range(len(transposed_data)):
            f_stat, p_value = f_oneway(*transposed_data[i])
            # record raw p-value
            p_values.append(p_value)
            # determine significance for this value
            significance.append(int(p_value < tr.common.get_configs('p_value')))
        # return raw p-values and binary flags for significance for output
        return [p_values, significance]

    def twoway_anova_kp(self, signal1, signal2, signal3, output_console=True, label_str=None):
        """Perform twoway ANOVA on 2 independent variables and 1 dependent variable (as list of lists).

        Args:
            signal1 (list): independent variable 1.
            signal2 (list): independent variable 2.
            signal3 (list of lists): dependent variable 1 (keypress data).
            output_console (bool, optional): whether to print results to console.
            label_str (str, optional): label to add before console output.

        Returns:
            df: results of ANOVA
        """
        # prepare signal1 and signal2 to be of the same dimensions as signal3
        signal3_flat = [value for sublist in signal3 for value in sublist]
        # number of observations in the dependent variable
        n_observations = len(signal3_flat)
        # repeat signal1 and signal2 to match the length of signal3_flat
        signal1_expanded = np.tile(signal1, n_observations // len(signal1))
        signal2_expanded = np.tile(signal2, n_observations // len(signal2))
        # create a datafarme with data
        data = pd.DataFrame({'signal1': signal1_expanded,
                             'signal2': signal2_expanded,
                             'dependent': signal3_flat
                             })
        # perform two-way ANOVA
        model = ols('dependent ~ C(signal1) + C(signal2) + C(signal1):C(signal2)', data=data).fit()
        anova_results = anova_lm(model)
        # print results to console
        if output_console and not label_str:
            print('Results for two-way ANOVA:\n', anova_results.to_string())
        if output_console and label_str:
            print('Results for two-way ANOVA for ' + label_str + ':\n', anova_results.to_string())
        return anova_results

    def save_stats_csv(self, t, p_values, name_file):
        """Save results of statistical test in csv.

        Args:
            t (list): list of time slices.
            p_values (list): list of p values.
            name_file (str): name of file.
        """
        path = os.path.join(tr.settings.output_dir, self.folder_stats)  # where to save csv
        # build path
        if not os.path.exists(path):
            os.makedirs(path)
        df = pd.DataFrame(columns=['t', 'p-value'])  # dataframe to save to csv
        df['t'] = t
        df['p-value'] = p_values
        df.to_csv(os.path.join(path, name_file))

    # def draw_ttest_anova(self, fig, times, name_file, yaxis_range, yaxis_step, ttest_signals, ttest_marker,
    #                      ttest_marker_size, ttest_marker_colour, ttest_annotations_font_size, ttest_annotations_colour,
    #                      anova_signals, anova_marker, anova_marker_size, anova_marker_colour,
    #                      anova_annotations_font_size, anova_annotations_colour, ttest_anova_row_height):
    #     """Draw ttest and anova test rows.

    #     Args:
    #         fig (figure): figure object.
    #         name_file (str): name of file to save.
    #         yaxis_range (list): range of x axis in format [min, max] for the keypress plot.
    #         yaxis_step (int): step between ticks on y axis.
    #         ttest_signals (list): signals to compare with ttest. None = do not compare.
    #         ttest_marker (str): symbol of markers for the ttest.
    #         ttest_marker_size (int): size of markers for the ttest.
    #         ttest_marker_colour (str): colour of markers for the ttest.
    #         ttest_annotations_font_size (int): font size of annotations for ttest.
    #         ttest_annotations_colour (str): colour of annotations for ttest.
    #         anova_signals (dict): signals to compare with ANOVA. None = do not compare.
    #         anova_marker (str): symbol of markers for the ANOVA.
    #         anova_marker_size (int): size of markers for the ANOVA.
    #         anova_marker_colour (str): colour of markers for the ANOVA.
    #         anova_annotations_font_size (int): font size of annotations for ANOVA.
    #         anova_annotations_colour (str): colour of annotations for ANOVA.
    #         ttest_anova_row_height (int): height of row of ttest/anova markers.
    #     """
    #     # count lines to calculate increase in coordinates of drawing
    #     counter_ttest = 0
    #     # count lines to calculate increase in coordinates of drawing
    #     counter_anova = 0
    #     # output ttest
    #     if ttest_signals:
    #         for signals in ttest_signals:
    #             # receive significance values
    #             [p_values, significance] = self.ttest(signal_1=signals['signal_1'],
    #                                                   signal_2=signals['signal_2'],
    #                                                   paired=signals['paired'])
    #             # save results to csv
    #             self.save_stats_csv(t=list(range(len(signals['signal_1']))),
    #                                 p_values=p_values,
    #                                 name_file=signals['label'] + '_' + name_file + '.csv')
    #             # add to the plot
    #             # plot stars based on random lists
    #             marker_x = []  # x-coordinates for stars
    #             marker_y = []  # y-coordinates for stars
    #             # assuming `times` and `signals['signal_1']` correspond to x and y data points
    #             for i in range(len(significance)):
    #                 if significance[i] == 1:  # if value indicates a star
    #                     marker_x.append(times[i])  # use the corresponding x-coordinate
    #                     # dynamically set y-coordinate, offset by ttest_anova_row_height for each signal_index
    #                     marker_y.append(-ttest_anova_row_height - counter_ttest * ttest_anova_row_height)
    #             # add scatter plot trace with cleaned data
    #             fig.add_trace(go.Scatter(x=marker_x,
    #                                      y=marker_y,
    #                                      # list of possible values: https://plotly.com/python/marker-style
    #                                      mode='markers',
    #                                      marker=dict(symbol=ttest_marker,  # marker
    #                                                  size=ttest_marker_size,  # adjust size
    #                                                  color=ttest_marker_colour),  # adjust colour
    #                                      text=p_values,
    #                                      showlegend=False,
    #                                      hovertemplate=signals['label'] + ': time=%{x}, p_value=%{text}'),
    #                           row=1,
    #                           col=1)
    #             # add label with signals that are compared
    #             fig.add_annotation(text=signals['label'],
    #                                # put labels at the start of the x axis, as they are likely no significant effects
    #                                # in the start of the trial
    #                                x=1,
    #                                # draw in the negative range of y axis
    #                                y=-ttest_anova_row_height - counter_ttest * ttest_anova_row_height,
    #                                showarrow=False,
    #                                font=dict(size=ttest_annotations_font_size, color=ttest_annotations_colour))
    #             # increase counter of lines drawn
    #             counter_ttest = counter_ttest + 1
    #     # output ANOVA
    #     if anova_signals:
    #         # if ttest was plotted, take into account for y of the first row or marker
    #         if counter_ttest > 0:
    #             counter_anova = counter_ttest
    #         # calculate for given signals one by one
    #         for signals in anova_signals:
    #             # receive significance values
    #             [p_values, significance] = self.anova(signals)
    #             # save results to csv
    #             self.save_stats_csv(t=list(range(len(signals['signals'][0]))),
    #                                 p_values=p_values,
    #                                 name_file=signals['label'] + '_' + name_file + '.csv')
    #             # add to the plot
    #             marker_x = []  # x-coordinates for stars
    #             marker_y = []  # y-coordinates for stars
    #             # assuming `times` and `signals['signal_1']` correspond to x and y data points
    #             for i in range(len(significance)):
    #                 if significance[i] == 1:  # if value indicates a star
    #                     marker_x.append(times[i])  # use the corresponding x-coordinate
    #                     # dynamically set y-coordinate, slightly offset for each signal_index
    #                     marker_y.append(-ttest_anova_row_height - counter_anova * ttest_anova_row_height)
    #             # add scatter plot trace with cleaned data
    #             fig.add_trace(go.Scatter(x=marker_x,
    #                                      y=marker_y,
    #                                      # list of possible values: https://plotly.com/python/marker-style
    #                                      mode='markers',
    #                                      marker=dict(symbol=anova_marker,  # marker
    #                                                  size=anova_marker_size,  # adjust size
    #                                                  color=anova_marker_colour),  # adjust colour
    #                                      text=p_values,
    #                                      showlegend=False,
    #                                      hovertemplate='time=%{x}, p_value=%{text}'),
    #                           row=1,
    #                           col=1)
    #             # add label with signals that are compared
    #             fig.add_annotation(text=signals['label'],
    #                                # put labels at the start of the x axis, as they are likely no significant effects
    #                                # in the start of the trial
    #                                x=1,
    #                                # draw in the negative range of y axis
    #                                y=-ttest_anova_row_height - counter_anova * ttest_anova_row_height,
    #                                showarrow=False,
    #                                font=dict(size=anova_annotations_font_size, color=anova_annotations_colour))
    #             # increase counter of lines drawn
    #             counter_anova = counter_anova + 1
    #     # hide ticks of negative values on y axis assuming that ticks are at step of 5
    #     r = range(0, fig.layout['yaxis']['range'][1] + 1, yaxis_step)
    #     fig.update_layout(yaxis={'tickvals': list(r), 'ticktext': [t if t >= 0 else '' for t in r]})

    def draw_ttest_anova_from_files(self, fig, stim, times, name_file, yaxis_range, yaxis_step, 
                                anova_results_file, nonparametric_results_file,
                                within_marker, within_marker_size, within_marker_colour, 
                                between_marker, between_marker_size, between_marker_colour, 
                                interaction_marker, interaction_marker_size, interaction_marker_colour,
                                anova_annotations_font_size, anova_annotations_colour, 
                                within_ego0_marker, within_ego0_marker_size, within_ego0_marker_colour, 
                                within_ego1_marker, within_ego1_marker_size, within_ego1_marker_colour,
                                within_target0_marker, within_target0_marker_size, within_target0_marker_colour,
                                within_target1_marker, within_target1_marker_size, within_target1_marker_colour,
                                comp_mixed_marker,comp_mixed_marker_size,comp_mixed_marker_colour,
                                comp_simple_marker,comp_simple_marker_size,comp_simple_marker_colour,
                                ttest_anova_row_height):
        """
        Draw t-test and ANOVA test rows using precomputed results from files, including F and p-value annotations.

        Args:
            fig: Plotly figure object.
            times: List of time points for the x-axis.
            name_file: Name of the plot file.
            yaxis_range: Range of y-axis for keypress plots.
            yaxis_step: Step size for y-axis ticks.
            ttest_results_file: Path to CSV file with t-test results.
            anova_results_file: Path to CSV file with ANOVA results.
            ...
        """
        logger.info(f"Loading ANOVA results for stimulus {stim}")
        anova_results_file = os.path.join(
            tr.settings.output_dir, 'statistics', f'batch_{stim}_keypress_data_rank_anova_results.csv'
        )
        nonparametric_results_file = os.path.join(
            tr.settings.output_dir, 'statistics', f'batch_{stim}_keypress_data_nonparametric_results.csv'
        )
        if not os.path.exists(anova_results_file):
            logger.warning(f"ANOVA results file not found: {anova_results_file}")
            return
        # counter_ttest = 0
        counter_within = 1
        counter_between = 2
        counter_interaction = 3
        counter_within_ego0 = 4
        counter_within_ego1 = 5
        counter_within_target0 = 6
        counter_within_target1 = 7
        counter_comp_mixed=8
        counter_comp_simple=9

        # # Load t-test results
        # if ttest_results_file:
        #     ttest_results = pd.read_csv(ttest_results_file)
        #     for _, row in ttest_results.iterrows():
        #         significance = row['p-value'] < 0.05
        #         marker_x = [times[int(row['TimeIndex'])]] if significance else []
        #         marker_y = [-ttest_anova_row_height - counter_ttest * ttest_anova_row_height]
        #         fig.add_trace(go.Scatter(
        #             x=marker_x, y=marker_y,
        #             mode='markers',
        #             marker=dict(symbol=ttest_marker, size=ttest_marker_size, color=ttest_marker_colour),
        #             text=row['p-value'],
        #             hovertemplate=f"{row['label']}: p-value=%{{text}}",
        #             showlegend=False
        #         ))
        #         counter_ttest += 1
    # Load Non-Parametric Results
        if nonparametric_results_file and os.path.exists(nonparametric_results_file):
            nonparametric_results = pd.read_csv(nonparametric_results_file)
            significant_results = nonparametric_results[nonparametric_results['p-value'] < 0.05]

            for _, row in significant_results.iterrows():
                time_index = int(row['TimeIndex'])
                x_coord = times[time_index]
                comparison = row['Comparison']
                direction_text = row['Direction']
                hover_text = (
                    f"Comparison: {comparison}<br>"
                    f"Statistic: {row['Statistic']:.2f}<br>"
                    f"p-value: {row['p-value']:.3f}<br>"
                    f"Direction: {direction_text}"
                )
                # Define marker properties for each comparison
                if comparison == "Within Ego=0: TargetCar=1 vs TargetCar=0":
                    marker_y = [-ttest_anova_row_height * counter_within_ego0]
                    marker_style = dict(symbol=within_ego0_marker, size=within_ego0_marker_size, color=within_ego0_marker_colour)
                    annotation_text = "Within Ego=0"
                elif comparison == "Within Ego=1: TargetCar=1 vs TargetCar=0":
                    marker_y = [-ttest_anova_row_height * counter_within_ego1]
                    marker_style = dict(symbol=within_ego1_marker, size=within_ego1_marker_size, color=within_ego1_marker_colour)
                    annotation_text = "Within Ego=1"
                elif comparison == "Within Target=0: EgoCar=1 vs EgoCar=0":
                    marker_y = [-ttest_anova_row_height * counter_within_target0]
                    marker_style = dict(symbol=within_target0_marker, size=within_target0_marker_size, color=within_target0_marker_colour)
                    annotation_text = "Within Target=0"
                elif comparison == "Within Target=1: EgoCar=1 vs EgoCar=0":
                    marker_y = [-ttest_anova_row_height * counter_within_target1]
                    marker_style = dict(symbol=within_target1_marker, size=within_target1_marker_size, color=within_target1_marker_colour)
                    annotation_text = "Within Target=1"
                elif comparison == "Ego=1,Target=0 vs Ego=0,Target=1":
                    marker_y = [-ttest_anova_row_height * counter_comp_mixed]
                    marker_style = dict(symbol=comp_mixed_marker, size=comp_mixed_marker_size, color=comp_mixed_marker_colour)
                    annotation_text = "comp mixed"
                elif comparison == "Ego=1,Target=1 vs Ego=0,Target=0":
                    marker_y = [-ttest_anova_row_height * counter_comp_simple]
                    marker_style = dict(symbol=comp_simple_marker, size=comp_simple_marker_size, color=comp_simple_marker_colour)
                    annotation_text = "comp simple"
                else:
                    continue

                # Add marker to the plot
                fig.add_trace(go.Scatter(
                    x=[x_coord],
                    y=marker_y,
                    mode='markers',
                    marker=marker_style,
                    text=hover_text,  # Preformatted hover text
                    hovertemplate="%{text}<br>Time: %{x:.2f}s<extra></extra>",  # Dynamic hover template
                    showlegend=False
                ))

                # Add row title annotation
                fig.add_annotation(
                    text=annotation_text,
                    x=-1,
                    y=marker_y[0],
                    showarrow=False,
                    font=dict(size=anova_annotations_font_size, color=anova_annotations_colour),
                    align='left'
                )
                # # Plot Within Ego=0: TargetCar=1 vs TargetCar=0
                # if comparison == "Within Ego=0: TargetCar=1 vs TargetCar=0":
                #     fig.add_trace(go.Scatter(
                #         x=[x_coord],
                #         y=[-ttest_anova_row_height * counter_within_ego0],
                #         mode='markers',
                #         marker=dict(symbol=within_ego0_marker, size=within_ego0_marker_size, color=within_ego0_marker_colour),
                #         text=(
                #             f"{comparison}<br>"
                #             # f"Group MDV MeanRank: {row['Group0_MeanRank']:.2f}, SD: {row['SD_Group1']:.2f}, N: {row['N_Group1']}<br>"
                #             # f"Group AV MeanRank: {row['Group1_MeanRank']:.2f}, SD: {row['SD_Group2']:.2f}, N: {row['N_Group2']}<br>"
                #             f"U={row['Statistic']:.2f}, p={row['p-value']:.3f}, Direction: {row['Direction']}, Direction: {direction_text}"
                #         ),
                #         hovertemplate=(
                #             "<b>Within Ego=0</b><br>"
                #             "Time: %{x:.2f}s<br>"
                #             "Statistic: %{text}<extra></extra>"
                #             "%{text}<br>"
                #             "Direction: %{text}<extra></extra>"
                #         ),
                #         showlegend=False
                #     ))
                #     fig.add_annotation(
                #         text="Within Ego=0",
                #         x=-1,
                #         y=-ttest_anova_row_height * counter_within_ego0,
                #         showarrow=False,
                #         font=dict(size=anova_annotations_font_size, color=anova_annotations_colour),
                #         align='left'
                #     )

                # # Plot Within Ego=1: TargetCar=1 vs TargetCar=0
                # elif comparison == "Within Ego=1: TargetCar=1 vs TargetCar=0":
                #     fig.add_trace(go.Scatter(
                #         x=[x_coord],
                #         y=[-ttest_anova_row_height * counter_within_ego1],
                #         mode='markers',
                #         marker=dict(symbol=within_ego1_marker, size=within_ego1_marker_size, color=within_ego1_marker_colour),
                #         text=(
                #             f"{comparison}<br>"
                #             # f"Group1 Mean: {row['Mean_Group1']:.2f}, SD: {row['SD_Group1']:.2f}, N: {row['N_Group1']}<br>"
                #             # f"Group2 Mean: {row['Mean_Group2']:.2f}, SD: {row['SD_Group2']:.2f}, N: {row['N_Group2']}<br>"
                #             f"U={row['Statistic']:.2f}, p={row['p-value']:.3f}, Direction: {direction_text}"
                #         ),
                #         hovertemplate=(
                #             "<b>Within Ego=1</b><br>"
                #             "Time: %{x:.2f}s<br>"
                #             "Statistic: %{text}<extra></extra>"
                #             "p-value: %{text}<br>"
                #             "Direction: %{text}<extra></extra>"
                #         ),
                #         showlegend=False
                #     ))
                #     fig.add_annotation(
                #         text="Within Ego=1",
                #         x=-1,
                #         y=-ttest_anova_row_height * counter_within_ego1,
                #         showarrow=False,
                #         font=dict(size=anova_annotations_font_size, color=anova_annotations_colour),
                #         align='left'
                #     )

                # # Plot Within Target=0: EgoCar=1 vs EgoCar=0
                # elif comparison == "Within Target=0: EgoCar=1 vs EgoCar=0":
                #     fig.add_trace(go.Scatter(
                #         x=[x_coord],
                #         y=[-ttest_anova_row_height * counter_within_target0],
                #         mode='markers',
                #         marker=dict(symbol=within_target0_marker, size=within_target0_marker_size, color=within_target0_marker_colour),
                #         text=(
                #             f"{comparison}<br>"
                #             # f"Group1 Mean: {row['Mean_Group1']:.2f}, SD: {row['SD_Group1']:.2f}, N: {row['N_Group1']}<br>"
                #             # f"Group2 Mean: {row['Mean_Group2']:.2f}, SD: {row['SD_Group2']:.2f}, N: {row['N_Group2']}<br>"
                #             f"U={row['Statistic']:.2f}, p={row['p-value']:.3f}, Direction: {direction_text}"
                #         ),
                #         hovertemplate=(
                #             "<b>Within Ego=0</b><br>"
                #             "Time: %{x:.2f}s<br>"
                #             "Statistic: %{text}<extra></extra>"
                #             "p-value: %{text}<br>"
                #             "Direction: %{text}<extra></extra>"
                #         ),
                #         showlegend=False
                #     ))
                #     fig.add_annotation(
                #         text="Within Target=0",
                #         x=-1,
                #         y=-ttest_anova_row_height * counter_within_target0,
                #         showarrow=False,
                #         font=dict(size=anova_annotations_font_size, color=anova_annotations_colour),
                #         align='left'
                #     )

                # # Plot Within Target=1: EgoCar=1 vs EgoCar=0
                # elif comparison == "Within Target=1: EgoCar=1 vs EgoCar=0":
                #     fig.add_trace(go.Scatter(
                #         x=[x_coord],
                #         y=[-ttest_anova_row_height * counter_within_target1],
                #         mode='markers',
                #         marker=dict(symbol=within_target1_marker, size=within_target1_marker_size, color=within_target1_marker_colour),
                #         text=(
                #             f"{comparison}<br>"
                #             # f"Group1 Mean: {row['Mean_Group1']:.2f}, SD: {row['SD_Group1']:.2f}, N: {row['N_Group1']}<br>"
                #             # f"Group2 Mean: {row['Mean_Group2']:.2f}, SD: {row['SD_Group2']:.2f}, N: {row['N_Group2']}<br>"
                #             f"U={row['Statistic']:.2f}, p={row['p-value']:.3f}, Direction: {direction_text}"
                #         ),
                #         hovertemplate=(
                #             "<b>Within Target=1</b><br>"
                #             "Time: %{x:.2f}s<br>"
                #             "Statistic: %{text}<extra></extra>"
                #             "p-value: %{text}<br>"
                #             "Direction: %{text}<extra></extra>"
                #         ),
                #         showlegend=False
                #     ))
                #     fig.add_annotation(
                #         text="Within Target=1",
                #         x=-1,
                #         y=-ttest_anova_row_height * counter_within_target1,
                #         showarrow=False,
                #         font=dict(size=anova_annotations_font_size, color=anova_annotations_colour),
                #         align='left'
                #     )

        else:
            logger.warning(f"Non-parametric results file not found: {nonparametric_results_file}")

        # Update axis and layout
        fig.update_yaxes(range=yaxis_range, tickvals=list(range(0, yaxis_range[1], yaxis_step)))

        # Load ANOVA results
        if anova_results_file:
            anova_results = pd.read_csv(anova_results_file)
            for _, row in anova_results.iterrows():
                time_index = int(row['TimeIndex'])
                x_coord = times[time_index]
                
                # Plot within-group p-value marker
                if row['Within-TargetCar-p'] < 0.05:  # Significant marker
                    fig.add_trace(go.Scatter(
                        x=[x_coord],
                        y=[-ttest_anova_row_height * counter_within],  # First row
                        mode='markers',
                        marker=dict(symbol=within_marker, size=within_marker_size, color=within_marker_colour),
                        text=f"F={row['Within-TargetCar-F']:.2f}, p={row['Within-TargetCar-p']:.3f}",  # Add both F and p values
                        hovertemplate=(
                            "<b>Within Group</b><br>""Time: %{:.2f}s<br>"
                            "F-value: {:.2f}<br>"
                            "p-value: {:.3f}"
                        ).format(x_coord, row['Within-TargetCar-F'], row['Within-TargetCar-p']),
                        showlegend=False
                    ))
                        # Add row title annotation
                    fig.add_annotation(
                        text="Within",
                        x=-1,  # Place annotation at the start of the x-axis
                        y=-ttest_anova_row_height * counter_within,
                        showarrow=False,
                        font=dict(size=anova_annotations_font_size, color=anova_annotations_colour),
                        align='left'
                    )

                # Plot between-group p-value marker
                if row['Between-EgoCar-p'] < 0.05:  # Significant marker
                    fig.add_trace(go.Scatter(
                        x=[x_coord],
                        y=[-ttest_anova_row_height * counter_between],  # Second row
                        mode='markers',
                        marker=dict(symbol=between_marker, size=between_marker_size, color=between_marker_colour),
                        text=f"F={row['Between-EgoCar-F']:.2f}, p={row['Between-EgoCar-p']:.3f}",  # Add both F and p values
                        hovertemplate=(
                            "<b>Between Group</b><br>""Time: %{:.2f}s<br>"
                            "F-value: {:.2f}<br>"
                            "p-value: {:.3f}"
                        ).format(x_coord, row['Between-EgoCar-F'], row['Between-EgoCar-p']),
                        showlegend=False
                    ))
                        # Add row title annotation
                    fig.add_annotation(
                        text="Between",
                        x=-1,  # Place annotation at the start of the x-axis
                        y=-ttest_anova_row_height * counter_between,
                        showarrow=False,
                        font=dict(size=anova_annotations_font_size, color=anova_annotations_colour),
                        align='left'
                    )

                # Plot interaction p-value marker
                if row['Interaction-p'] < 0.05:  # Significant marker
                    fig.add_trace(go.Scatter(
                        x=[x_coord],
                        y=[-ttest_anova_row_height * counter_interaction],  # Third row
                        mode='markers',
                        marker=dict(symbol=interaction_marker, size=interaction_marker_size, color=interaction_marker_colour),
                        text=f"F={row['Interaction-F']:.2f}, p={row['Interaction-p']:.3f}",  # Add both F and p values
                        hovertemplate=(
                            "<b>Interaction</b><br>""Time: %{:.2f}s<br>"
                            "F-value: {:.2f}<br>"
                            "p-value: {:.3f}"
                        ).format(x_coord, row['Interaction-F'], row['Interaction-p']),
                        showlegend=False
                    ))
                        # Add row title annotation
                    fig.add_annotation(
                        text="Interaction",
                        x=-1,  # Place annotation at the start of the x-axis
                        y=-ttest_anova_row_height * counter_interaction,
                        showarrow=False,
                        font=dict(size=anova_annotations_font_size, color=anova_annotations_colour),
                        align='left'
                    )



        # Update axis and layout
        fig.update_yaxes(range=yaxis_range, tickvals=list(range(0, yaxis_range[1], yaxis_step)))


    def draw_events(self, fig, yaxis_range, events, events_width, events_dash, events_colour,
                    events_annotations_font_size, events_annotations_colour):
        """Draw lines and annotations of events.

        Args:
            fig (figure): figure object.
            yaxis_range (list): range of x axis in format [min, max] for the keypress plot.
            events (list): list of events to draw formatted as values on x axis.
            events_width (int): thickness of the vertical lines.
            events_dash (str): type of the vertical lines.
            events_colour (str): colour of the vertical lines.
            events_annotations_font_size (int): font size of annotations for the vertical lines.
            events_annotations_colour (str): colour of annotations for the vertical lines.
        """
        # count lines to calculate increase in coordinates of drawing
        counter_lines = 0
        # draw lines with annotations for events
        if events:
            for event in events:
                # draw start
                fig.add_shape(type='line',
                              x0=event['start'],
                              y0=0,
                              x1=event['start'],
                              y1=yaxis_range[1] - counter_lines * 2 - 2,
                              line=dict(color=events_colour,
                                        dash=events_dash,
                                        width=events_width))
                # draw other elements only is start and finish are not the same
                if event['start'] != event['end']:
                    # draw finish
                    fig.add_shape(type='line',
                                  x0=event['end'],
                                  y0=0,
                                  x1=event['end'],
                                  y1=yaxis_range[1] - counter_lines * 2 - 2,
                                  line=dict(color=events_colour,
                                            dash=events_dash,
                                            width=events_width))
                    # draw horizontal line
                    fig.add_annotation(ax=event['start'],
                                       axref='x',
                                       ay=yaxis_range[1] - counter_lines * 2 - 2,
                                       ayref='y',
                                       x=event['end'],
                                       arrowcolor='black',
                                       xref='x',
                                       y=yaxis_range[1] - counter_lines * 2 - 2,
                                       yref='y',
                                       arrowwidth=events_width,
                                       arrowside='end+start',
                                       arrowsize=1,
                                       arrowhead=2)
                    # draw text label
                    fig.add_annotation(text=event['annotation'],
                                       x=(event['end'] + event['start']) / 2,
                                       y=yaxis_range[1] - counter_lines * 2 - 1,  # use ylim value and draw lower
                                       showarrow=False,
                                       font=dict(size=events_annotations_font_size, color=events_annotations_colour))
                # increase counter of lines drawn
                counter_lines = counter_lines + 2
