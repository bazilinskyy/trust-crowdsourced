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
import warnings
import unicodedata
import re
from tqdm import tqdm
import ast
from scipy.stats.kde import gaussian_kde
# from PIL import Image
import cv2
import trust as tr
from statistics import mean

matplotlib.use('TkAgg')
logger = tr.CustomLogger(__name__)  # use custom logger


class Analysis:
    # set template for plotly output
    template = tr.common.get_configs('plotly_template')
    # store resolution for keypress data
    res = tr.common.get_configs('kp_resolution')
    # number of stimuli
    num_stimuli = tr.common.get_configs('num_stimuli')
    # folder for output
    fig = None
    g = None
    image = None
    stim_id = None
    points = None
    save_frames = False
    folder = '/figures/'
    polygons = None

    def __init__(self):
        # set font to Times
        plt.rc('font', family='serif')

    def save_all_frames(self, df, mapping, id_video, t):
        """
        Outputs individual frames as png from inputted video mp4.

        Args:
            df (dataframe): dataframe of heroku.
            id_video (int): stimulus video ID.
            t (list): column in dataframe containing time data.
        """
        logger.info('Creating frames')
        # path for temp folder to store images with frames
        path = os.path.join(tr.settings.output_dir, 'frames')
        # create temp folder
        if not os.path.exists(path):
            os.makedirs(path)
        # video file in the folder with stimuli
        cap = cv2.VideoCapture(os.path.join(tr.common.get_configs('path_stimuli'),  # noqa: E501
                                            'video_' + str(id_video) + '.mp4'))  # noqa: E501
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
            cap.set(cv2.CAP_PROP_POS_FRAMES, round(fps * k/1000))
            ret, frame = cap.read()
            if ret:
                filename = os.path.join(path,
                                        'frame_' + str([round(k/hm_resolution_int)]) + '.jpg')  # noqa: E501
                cv2.imwrite(filename,
                            frame,
                            [cv2.IMWRITE_JPEG_QUALITY, 20])

    def create_histogram(self,
                         image,
                         points,
                         id_video,
                         density_coef=10,
                         suffix='_histogram.jpg',
                         save_file=False):
        """
        Create histogram for image based on the list of lists of points.
        density_coef: coeficient for division of dimensions for density of
                        points.
        """
        # check if data is present
        if not points:
            logger.error('Not enough data. Histogram was not created for {}.',
                         image)
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
        fig = plt.figure(figsize=(width/dpi, height/dpi), dpi=dpi)
        # build histogram
        plt.hist2d(x=x,
                   y=-y,  # convert to the reference system in image
                   bins=[round(width/density_coef),
                         round(height/density_coef)],
                   cmap=plt.cm.jet)
        plt.colorbar()
        # remove white spaces around figure
        plt.gca().set_axis_off()
        # save image
        if save_file:
            self.save_fig(image, fig,
                          self.folder, '_video_' + str(id_video)+suffix)

    def create_heatmap(self,
                       image,
                       points,
                       type_heatmap='contourf',
                       add_corners=True,
                       save_file=False):
        """
        Create heatmap for image based on the list of lists of points.
        add_corners: add points to the corners to have the heatmap ovelay the
                     whole image
        type_heatmap: contourf, pcolormesh, kdeplot
        """
        # todo: remove datapoints in corners in heatmaps
        # check if data is present
        if not points:
            logger.error('Not enough data. Heatmap was not created for {}.',
                         image)
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
            xi, yi = np.mgrid[x.min():x.max():x.size**0.5*1j,
                              y.min():y.max():y.size**0.5*1j]
            zi = k(np.vstack([xi.flatten(), yi.flatten()]))
        except (np.linalg.LinAlgError, np.linalg.LinAlgError, ValueError):
            logger.error('Not enough data. Heatmap was not created for {}.',
                         image)
            return
        # create figure object with given dpi and dimensions
        dpi = 150
        fig = plt.figure(figsize=(width/dpi, height/dpi), dpi=dpi)
        # alpha=0.5 makes the plot semitransparent
        suffix_file = ''  # suffix to add to saved image
        if type_heatmap == 'contourf':
            try:
                g = plt.contourf(xi, yi, zi.reshape(xi.shape),
                                 alpha=0.5)
                plt.margins(0, 0)
                plt.gca().xaxis.set_major_locator(plt.NullLocator())
                plt.gca().yaxis.set_major_locator(plt.NullLocator())
            except TypeError:
                logger.error('Not enough data. Heatmap was not created for '
                             + '{}.',
                             image)
                plt.close(fig)  # clear figure from memory
                return
            suffix_file = '_contourf.jpg'
        elif type_heatmap == 'pcolormesh':
            try:
                g = plt.pcolormesh(xi, yi, zi.reshape(xi.shape),
                                   shading='auto',
                                   alpha=0.5)
                plt.margins(0, 0)
                plt.gca().xaxis.set_major_locator(plt.NullLocator())
                plt.gca().yaxis.set_major_locator(plt.NullLocator())
            except TypeError:
                logger.error('Not enough data. Heatmap was not created for '
                             + '{}.',
                             image)
                plt.close(fig)  # clear figure from memory
                return
            suffix_file = '_pcolormesh.jpg'
        elif type_heatmap == 'kdeplot':
            try:
                g = sns.kdeplot(x=x,
                                y=y,
                                alpha=0.5,
                                fill=True,
                                cmap="RdBu_r")
            except TypeError:
                logger.error('Not enough data. Heatmap was not created for '
                             + '{}.',
                             image)
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
        plt.subplots_adjust(top=1,
                            bottom=0,
                            right=1,
                            left=0,
                            hspace=0,
                            wspace=0)
        # save image
        if save_file:
            self.save_fig(image, fig, '/figures/', suffix_file)
        # return graph objects
        return fig, g

    def create_animation(self,
                         df,
                         mapping,
                         image,
                         id_video,
                         points,
                         t,
                         save_anim=False,
                         save_frames=False):
        """
        Create animation for image based on the list of lists of points of
        varying duration.
        image =  the frames from the stimulus video
        id_video = which stimulus video is being used
        dur = duration of given video stimulus
        self.framess = the ammount of frames that fit into
            given stimmulus video based on resolution
        self.points = list containting eye-tracking points
        self.t = time
        self.times = time
        self.fig = figure
        self.kp_data = keypress data for given stimulus video

        """
        self.image = image
        self.id_video = id_video
        # calc amounts of steps from duration
        dur = mapping.iloc[id_video]['video_length']
        # Determing the amount of frames for given video
        self.framess = int(round(self.time/self.hm_resolution))
        # Determin time
        self.t = mapping.loc['video_'+str(id_video)][t]
        # Call eye-tracking points
        self.points = points
        self.save_frames = save_frames
        # Create subplot figure with heatmap and kp plot
        self.fig, self.g = plt.subplots(nrows=2,
                                        ncols=1,
                                        figsize=(20, 10),
                                        gridspec_kw=dict(height_ratios=[1, 2],
                                                         hspace=0.2))
        self.fig.suptitle(' Keypresses and eye-tracking heatmap ', fontsize=20)
        # Deterin time and data for kp plot
        self.times = np.array(range(self.res,
                              mapping['video_length'].max() + self.res,
                              self.res)) / 1000
        self.kp_data = mapping.loc['video_' + str(id_video)]['kp']
        # Animate frames subplots into one animation using animate function
        anim = animation.FuncAnimation(self.fig,
                                       self.animate,
                                       frames=self.framess,
                                       interval=self.hm_resolution,
                                       repeat=False)
        # save image
        if save_anim:
            self.save_anim(image, anim, self.folder,
                           '_video_' + str(id_video) + '_animation.mp4')

    def create_animation_all_stimuli(self, num_stimuli):
        """
        Create long video with all animations.
        """
        logger.info('Creating long video with all animations for {} stimuli.',
                    num_stimuli)
        # create path
        path = tr.settings.output_dir + self.folder
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
        """
        self.g[1].clear()
        durations = range(self.hm_resolution)
        # KDE plot data
        it = int(round(len(self.kp_data)*i/(self.framess)))
        print(it)
        self.g[0].plot(np.array(self.times[:it]),
                       np.array(self.kp_data[:it]),
                       lw=1,
                       color='r')

        self.g[0].set_xlabel("Time")
        self.g[0].set_ylabel("number Keypresses")
        self.g[0].set_xlim(0, 50)
        self.g[0].set_ylim(0, 50)
        self.g[0].set_title('id_video=' + str(self.id_video))
        self.g[1] = sns.kdeplot(x=[item[0] for item in self.points[i]],
=======

        length = int(len(self.event))
        for ev in range(len(self.event)):
            self.g[0].axvline(x=int(self.event[ev])/1000,
                              label="" + str(self.event_discription[ev]),
                              c=plt.cm.RdYlBu(int(ev)/length),
                              lw=2)
            self.g[0].tick_params(axis='x', labelrotation=90)
            self.g[0].legend()
            self.g[1].axvline(x=int(self.event[ev])/1000,
                              label="" + str(self.event_discription[ev]),
                              c=plt.cm.RdYlBu(int(ev)/length),
                              lw=2)
            self.g[1].tick_params(axis='x', labelrotation=90)
            self.g[1].legend()
        # Subplot 2 AOI
        self.g[1].set_title('Number of eye gazes in area of interest', fontsize=25)  # noqa: E501
        self.g[1].set_xlabel('Time (s)', fontsize=10)
        self.g[1].set_ylabel('Number of gazes in Area of Interest', fontsize=10)  # noqa: E501
        self.g[1].set_xlim(0, 50)
        self.g[1].set_ylim(0, 20)
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
        x = [item[0] for item in self.points[i]]
        y = [item[1] for item in self.points[i]]
        # stim 21 - 41
        x1 = [item[0] for item in self.points1[i]]
        y1 = [item[1] for item in self.points1[i]]
        # stim 42 - 62
        x2 = [item[0] for item in self.points2[i]]
        y2 = [item[1] for item in self.points2[i]]
        # stim 63 - 83
        x3 = [item[0] for item in self.points3[i]]
        y3 = [item[1] for item in self.points3[i]]
        # Filtering data for if they are inside or outside coordinates
        num = 0
        num1 = 0
        num2 = 0
        num3 = 0
        for v in range(len(x)):
            if max_x > x[v] > min_x:
                if max_y > y[v] > min_y:
                    num = num + 1
                else:
                    continue
            else:
                continue
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

        self.number_in.append(int(num))
        self.number_in1.append(int(num1))
        self.number_in2.append(int(num2))
        self.number_in3.append(int(num3))
        self.g[1].plot(self.aoit,
                       self.number_in,
                       label='Stim 1',
                       color='r')
        self.g[1].plot(self.aoit,
                       self.number_in1,
                       label='Stim 2',
                       color='b')
        self.g[1].plot(self.aoit,
                       self.number_in2,
                       label='stim 3',
                       color='g')
        self.g[1].plot(self.aoit,
                       self.number_in3,
                       label='Stim 4',
                       color='m')
        self.g[1].legend()
        # Subplot 3 Heatmap
        self.g[2] = sns.kdeplot(x=[item[0] for item in self.points[i]],
>>>>>>> Stashed changes
                                y=[item[1] for item in self.points[i]],
                                alpha=0.5,
                                fill=True,
                                cmap='RdBu_r')
        # Scatter plot data
        # 1 person
        # item1 = ([item[0] for item in self.points[i]])
        # item2 = ([item[1] for item in self.points[i]])
        # self.g = sns.scatterplot(x=item1[8],
        #                          y=item2[8],
        #                          alpha=0.5,
        #                          hue=[item[0] for item in self.points[i]],
        #                          legend='auto'
        #                          )
        # all pp
        # self.g = sns.scatterplot(x=[item[0] for item in self.points[i]],
        #                          y=[item[1] for item in self.points[i]],
        #                          alpha=0.5,
        #                          hue=[item[0] for item in self.points[i]],
        #                          legend='auto')
        # read original image
        im = plt.imread(self.image + '\\frame_' + str([i]) + '.jpg')
        plt.imshow(im)
        # remove axis
        plt.gca().set_axis_off()
        # remove white spaces around figure
        # plt.subplots_adjust(top=1,
        #                     bottom=0,
        #                     right=1,
        #                     left=0,
        #                     hspace=0,
        #                     wspace=0)
        # textbox with duration

        # props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        # plt.text(0.75,
        #          0.98,
        #          'id_video=' + str(self.id_video) +
        #          ' time (ms)=' + str(round(durations[i]*int(self.t)/self.hm_resolution)), # noqa: E501
        #          transform=plt.gca().transAxes,
        #          fontsize=12,
        #          verticalalignment='top',
        #          bbox=props)
        # save each frame as file
        if self.save_frames:
            # build suffix for filename
            suffix = '_kdeplot_' + str(durations[i]) + '.jpg'
            # copy figure in buffer to prevent distruction of object
            buf = io.BytesIO()
            pickle.dump(self.fig, buf)
            buf.seek(0)
            temp_fig = pickle.load(buf)
            # save figure
            self.save_fig(self.image, temp_fig, self.folder, suffix)
        return self.g
        # save each frame as file
        if self.save_frames:
            # build suffix for filename
            suffix = '_kdeplot_' + str(durations[i]) + '.jpg'
            # copy figure in buffer to prevent distruction of object
            buf = io.BytesIO()
            pickle.dump(self.fig, buf)
            buf.seek(0)
            temp_fig = pickle.load(buf)
            # save figure
            self.save_fig(self.image, temp_fig, self.folder, suffix)
        return self.g

    def save_anim(self, image, anim, output_subdir, suffix):
        """
        Helper function to save figure as file.
        """
        # extract name of stimulus after last slash
        file_no_path = image.rsplit('/', 1)[-1]
        # remove extension
        file_no_path = os.path.splitext(file_no_path)[0]
        # create path
        path = tr.settings.output_dir + output_subdir
        if not os.path.exists(path):
            os.makedirs(path)
        # save file
        anim.save(path + suffix, writer='ffmpeg')
        # clear animation from memory
        plt.close(self.fig)

    # TODO: fix error with value of string not used as float
    def corr_matrix(self, df, columns_drop, save_file=False):
        """
        Output correlation matrix.

        Args:
            df (dataframe): mapping dataframe.
            columns_drop (list): columns dataframes in to ignore.
            save_file (bool, optional): flag for saving an html file with plot.
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
        vs_font = 10  # very small
        s_font = 12   # small
        m_font = 16   # medium
        l_font = 18   # large
        plt.rc('font', size=s_font)         # controls default text sizes
        plt.rc('axes', titlesize=s_font)    # fontsize of the axes title
        plt.rc('axes', labelsize=s_font)   # fontsize of the axes labels
        plt.rc('xtick', labelsize=vs_font)   # fontsize of the tick labels
        plt.rc('ytick', labelsize=vs_font)   # fontsize of the tick labels
        plt.rc('legend', fontsize=s_font)   # fontsize of the legend
        plt.rc('figure', titlesize=l_font)  # fontsize of the figure title
        plt.rc('axes', titlesize=m_font)    # fontsize of the subplot title
        # create figure
        fig = plt.figure(figsize=(34, 20))
        g = sns.heatmap(corr,
                        annot=True,
                        mask=mask,
                        cmap='coolwarm',
                        fmt=".2f")
        # rotate ticks
        for item in g.get_xticklabels():
            item.set_rotation(55)
        # save image
        if save_file:
            self.save_fig('all',
                          fig,
                          self.folder,
                          '_corr_matrix.jpg',
                          pad_inches=0.05)
        # revert font
        self.reset_font()

    def scatter_matrix(self, df, columns_drop, color=None, symbol=None,
                       diagonal_visible=False, xaxis_title=None,
                       yaxis_title=None, save_file=False):
        """
        Output scatter matrix.

        Args:
            df (dataframe): mapping dataframe.
            columns_drop (list): columns dataframes in to ignore.
            color (str, optional): dataframe column to assign color of points.
            symbol (str, optional): dataframe column to assign symbol of
                                    points.
            diagonal_visible (bool, optional): show/hide diagonal with
                                               correlation==1.0.
            xaxis_title (str, optional): title for x axis.
            yaxis_title (str, optional): title for y axis.
            save_file (bool, optional): flag for saving an html file with plot.
        """
        logger.info('Creating scatter matrix.')
        # drop columns
        df = df.drop(columns=columns_drop)
        # create dimensions list after dropping columns
        dimensions = df.keys()
        # plot matrix
        fig = px.scatter_matrix(df,
                                dimensions=dimensions,
                                color=color,
                                symbol=symbol)
        # update layout
        fig.update_layout(template=self.template,
                          width=5000,
                          height=5000,
                          xaxis_title=xaxis_title,
                          yaxis_title=yaxis_title)
        # hide diagonal
        if not diagonal_visible:
            fig.update_traces(diagonal_visible=False)
        # save file
        if save_file:
            self.save_plotly(fig, 'scatter_matrix', self.folder)
        # open it in localhost instead
        else:
            fig.show()

    def bar(self, df, y: list, x=None, stacked=False, pretty_text=False,
            orientation='v', xaxis_title=None, yaxis_title=None,
            show_all_xticks=False, show_all_yticks=False,
            show_text_labels=False, save_file=False):
        """
        Barplot for questionnaire data. Passing a list with one variable will
        output a simple barplot; passing a list of variables will output a
        grouped barplot.

        Args:
            df (dataframe): dataframe with data from appen.
            x (list): values in index of dataframe to plot for. If no value is
                      given, the index of df is used.
            y (list): column names of dataframe to plot.
            stacked (bool, optional): show as stacked chart.
            pretty_text (bool, optional): prettify ticks by replacing _ with
                                          spaces and capitilisng each value.
            orientation (str, optional): orientation of bars. v=vertical,
                                         h=horizontal.
            xaxis_title (str, optional): title for x axis.
            yaxis_title (str, optional): title for y axis.
            show_all_xticks (bool, optional): show all ticks on x axis.
            show_all_yticks (bool, optional): show all ticks on y axis.
            show_text_labels (bool, optional): output automatically positionsed
                                               text labels.
            save_file (bool, optional): flag for saving an html file with plot.
        """
        logger.info('Creating bar chart for x={} and y={}', x, y)
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
        for variable in y:
            # showing text labels
            if show_text_labels:
                text = df[variable]
            else:
                text = None
            # plot variable
            fig.add_trace(go.Bar(x=x,
                                 y=df[variable],
                                 name=variable,
                                 orientation=orientation,
                                 text=text,
                                 textposition='auto'))
        # add tabs if multiple variables are plotted
        if len(y) > 1:
            fig.update_layout(barmode='group')
            buttons = list([dict(label='All',
                                 method='update',
                                 args=[{'visible': [True] * df[y].shape[0]},
                                       {'title': 'All',
                                        'showlegend': True}])])
            # counter for traversing through stimuli
            counter_rows = 0
            for variable in y:
                visibility = [[counter_rows == j] for j in range(len(y))]
                visibility = [item for sublist in visibility for item in sublist]  # type: ignore # noqa: E501
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
        fig.update_layout(template=self.template,
                          xaxis_title=xaxis_title,
                          yaxis_title=yaxis_title)
        # format text labels
        if show_text_labels:
            fig.update_traces(texttemplate='%{text:.2s}')
        # show all ticks on x axis
        if show_all_xticks:
            fig.update_layout(xaxis=dict(dtick=1))
        # show all ticks on x axis
        if show_all_yticks:
            fig.update_layout(yaxis=dict(dtick=1))
        # stacked bar chart
        if stacked:
            fig.update_layout(barmode='stack')
        # save file
        if save_file:
            file_name = 'bar_' + '-'.join(str(val) for val in y) + '_' + \
                        '-'.join(str(val) for val in x)
            self.save_plotly(fig,
                             file_name,
                             self.folder)
        # open it in localhost instead
        else:
            fig.show()

    def scatter(self, df, x, y, color=None, symbol=None, size=None, text=None,
                trendline=None, hover_data=None, marker_size=None,
                pretty_text=False, marginal_x='violin', marginal_y='violin',
                xaxis_title=None, yaxis_title=None, xaxis_range=None,
                yaxis_range=None, save_file=True):
        """
        Output scatter plot of variables x and y with optional assignment of
        colour and size.

        Args:
            df (dataframe): dataframe with data from heroku.
            x (str): dataframe column to plot on x axis.
            y (str): dataframe column to plot on y axis.
            color (str, optional): dataframe column to assign color of points.
            symbol (str, optional): dataframe column to assign symbol of
                                    points.
            size (str, optional): dataframe column to assign soze of points.
            text (str, optional): dataframe column to assign text labels.
            trendline (str, optional): trendline. Can be 'ols', 'lowess'
            hover_data (list, optional): dataframe columns to show on hover.
            marker_size (int, optional): size of marker. Should not be used
                                         together with size argument.
            pretty_text (bool, optional): prettify ticks by replacing _ with
                                          spaces and capitilisng each value.
            marginal_x (str, optional): type of marginal on x axis. Can be
                                        'histogram', 'rug', 'box', or 'violin'.
            marginal_y (str, optional): type of marginal on y axis. Can be
                                        'histogram', 'rug', 'box', or 'violin'.
            xaxis_title (str, optional): title for x axis.
            yaxis_title (str, optional): title for y axis.
            xaxis_range (list, optional): range of x axis in format [min, max].
            yaxis_range (list, optional): range of y axis in format [min, max].
            save_file (bool, optional): flag for saving an html file with plot.
        """
        logger.info('Creating scatter plot for x={} and y={}.', x, y)
        # using size and marker_size is not supported
        if marker_size and size:
            logger.error('Arguments marker_size and size cannot be used'
                         + ' together.')
            return -1
        # using marker_size with histogram marginal(s) is not supported
        if (marker_size and
                (marginal_x == 'histogram' or marginal_y == 'histogram')):
            logger.error('Argument marker_size cannot be used together with'
                         + ' histogram marginal(s).')
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
                logger.debug('Tried to prettify {} with exception {}', text, e)
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
        # save file
        if save_file:
            self.save_plotly(fig,
                             'scatter_' + x + '-' + y,
                             self.folder)
        # open it in localhost instead
        else:
            fig.show()

    def scatter_mult(self, df, x, y, color=None, symbol=None,
                     text=None, trendline=None, hover_data=None,
                     marker_size=None, pretty_text=False, marginal_x='violin',
                     marginal_y='violin', xaxis_title=None, yaxis_title=None,
                     xaxis_range=None, yaxis_range=None, save_file=True):
        """
        Output scatter plot of multiple variables x and y with optional
        assignment of colour and size.

        Args:
            df (dataframe): dataframe with data from heroku.
            x (str): dataframe columns to plot on x axis.
            y (str): dataframe column to plot on y axis.
            symbol (str, optional): dataframe column to assign symbol of
                                    points.
            text (str, optional): dataframe column to assign text labels.
            trendline (str, optional): trendline. Can be 'ols', 'lowess'
            hover_data (list, optional): dataframe columns to show on hover.
            marker_size (int, optional): size of marker. Should not be used
                                         together with size argument.
            pretty_text (bool, optional): prettify ticks by replacing _ with
                                          spaces and capitilisng each value.
            marginal_x (str, optional): type of marginal on x axis. Can be
                                        'histogram', 'rug', 'box', or 'violin'.
            marginal_y (str, optional): type of marginal on y axis. Can be
                                        'histogram', 'rug', 'box', or 'violin'.
            xaxis_title (str, optional): title for x axis.
            yaxis_title (str, optional): title for y axis.
            xaxis_range (list, optional): range of x axis in format [min, max].
            yaxis_range (list, optional): range of y axis in format [min, max].
            save_file (bool, optional): flag for saving an html file with plot.
        """
        # todo: extend with multiple columns for y
        logger.info('Creating scatter plot for x={} and y={}.', x, y)
        # using marker_size with histogram marginal(s) is not supported
        if (marker_size and
                (marginal_x == 'histogram' or marginal_y == 'histogram')):
            logger.error('Argument marker_size cannot be used together with'
                         + ' histogram marginal(s).')
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
                    logger.error('no string')

            if isinstance(df.iloc[0][y], str):  # check if string
                # replace underscores with spaces
                df[y] = df[y].str.replace('_', ' ')
                # capitalise
                df[y] = df[y].str.capitalize()
            else:
                logger.error('no string')
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
                                      x=0.78
                                      ))

        # results = px.get_trendline_results(fig)
        # for i in range(len(x)):
        #     print(results.px_fit_results.iloc[i].summary())
        # change marker size
        # if marker_size:
        #    fig.update_traces(marker=dict(size=marker_size))

        # change marker size
        if marker_size:
            fig.update_traces(marker=dict(size=marker_size))

        # save file
        if save_file:
            self.save_plotly(fig,
                             'scatter_' + ','.join(x) + '-' + y,
                             self.folder)
        # open it in localhost instead
        else:
            fig.show()

    # todo: @Job, add docstring
    def scatter_et(self, df, x, y, t, pp, id_video, pretty_text=False,
                   marginal_x='violin', marginal_y='violin', xaxis_title=None,
                   xaxis_range=True, yaxis_title=None, yaxis_range=True,
                   save_file=True):
        """
        output satter plot of x & y.

        Args:
            df (dataframe): dataframe with data from Heroku.
            x (list): dataframe column to plot on x axis.
            y (list): dataframe column to plot on y axis.
            t (list): dataframe column to determin timespan
            pp (int): participant ID.
            id_video (int): stimulus video ID.
            pretty_text (bool, optional): prettify ticks by replacing _ with
                                          spaces and capitalising each value.
            marginal_x (str, optional): type of marginal on x axis. Can be
                                        'histogram', 'rug', 'box', or 'violin'.
            marginal_y (str, optional): type of marginal on y axis. Can be
                                        'histogram', 'rug', 'box', or 'violin'.
            xaxis_range (list, optional): range of the x-axis plot.
            yaxis_range (list, optional): range of the y-axis plot.
            xaxis_title (str, optional): title for x axis.
            yaxis_title (str, optional): title for y axis.
            save_file (bool, optional): flag for saving an html file with plot.
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
                         title='scatter_' + ' ' + id_video + ' ' + 'participant' + ' ' + pp)  # noqa: E501

        # update layout
        fig.update_layout(template=self.template,
                          xaxis_title=xaxis_title,
                          yaxis_title=yaxis_title,
                          xaxis_range=[0, width],
                          yaxis_range=[0, height])

        # save file
        if save_file:
            self.save_plotly(fig,
                             'scatter_map_' + id_video+'_participant_' + pp,
                             self.folder)
        # open it in localhost instead
        else:
            fig.show()

    def heatmap(self, df, x, y, t, id_video, pp, pretty_text=False,
                marginal_x='violin', marginal_y='violin', xaxis_title=None,
                xaxis_range=True, yaxis_title=None, yaxis_range=True,
                save_file=True):
        """
        Output heatmap plot of variables x and y.

        Args:
            df (dataframe): dataframe with data from heroku.
            x (list): dataframe column to plot on x axis.
            y (list): dataframe column to plot on y axis.
            pretty_text (bool, optional): prettify ticks by replacing _ with
                                          spaces and capitalising each value.
            marginal_x (str, optional): type of marginal on x axis. Can be
                                        'histogram', 'rug', 'box', or 'violin'.
            marginal_y (str, optional): type of marginal on y axis. Can be
                                        'histogram', 'rug', 'box', or 'violin'.
            xaxis_title (str, optional): title for x axis.
            yaxis_title (str, optional): title for y axis.
            save_file (bool, optional): flag for saving an html file with plot.
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
                          title='heatmap_scatter_animation'+' ' + id_video + ' ' + 'participant'+' '+pp,  # noqa: E501
                          xaxis_range=[0, 2*width],
                          yaxis_range=[0, 2*height],
                          updatemenus=[dict(type='buttons',
                                            buttons=[dict(label='Play',
                                                          method='animate',
                                                          args=[None,
                                                                dict(fromcurrent=True,  # noqa: E501
                                                                     transition={'duration': 10},  # noqa: E501
                                                                     frame=dict(redraw=True,  # noqa: E501
                                                                                duration=100))]),  # noqa: E501
                                                     dict(label='Pause',
                                                          method='animate',
                                                          args=[[None],
                                                                dict(fromcurrent=True,  # noqa: E501
                                                                     mode='immediate',  # noqa: E501
                                                                     transition={'duration': 10},  # noqa: E501
                                                                     frame=dict(redraw=True,  # noqa: E501
                                                                                duration=100))])])])  # noqa: E501
        # save file
        if save_file:
            self.save_plotly(fig,
                             'heatmap_animation' + id_video+'_participant_' + pp,  # noqa: E501
                             self.folder)
        # open it in localhost instead
        else:
            # plotly.offline.plot(fig, auto_play = False)
            # TODO: error with show
            # show.fig(fig, auto_play=False)
            logger.error('show not implemented')

    def hist(self, df, x, nbins=None, color=None, pretty_text=False,
             marginal='rug', xaxis_title=None, yaxis_title=None,
             save_file=True):
        """
        Output histogram of time of participation.

        Args:
            df (dataframe): dataframe with data from heroku.
            x (list): column names of dataframe to plot.
            nbins (int, optional): number of bins in histogram.
            color (str, optional): dataframe column to assign color of circles.
            pretty_text (bool, optional): prettify ticks by replacing _ with
                                          spaces and capitilisng each value.
            marginal (str, optional): type of marginal on x axis. Can be
                                      'histogram', 'rug', 'box', or 'violin'.
            xaxis_title (str, optional): title for x axis.
            yaxis_title (str, optional): title for y axis.
            save_file (bool, optional): flag for saving an html file with plot.
        """
        logger.info('Creating histogram for x={}.', x)
        # using colour with multiple values to plot not supported
        if color and len(x) > 1:
            logger.error('Color property can be used only with a single' +
                         ' variable to plot.')
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
            fig = px.histogram(df[x], nbins=nbins, marginal=marginal,
                               color=df[color])
        else:
            fig = px.histogram(df[x], nbins=nbins, marginal=marginal)
        # ticks as numbers
        fig.update_layout(xaxis=dict(tickformat='digits'))
        # update layout
        fig.update_layout(template=self.template,
                          xaxis_title=xaxis_title,
                          yaxis_title=yaxis_title)
        # save file
        if save_file:
            self.save_plotly(fig,
                             'hist_' + '-'.join(str(val) for val in x),
                             self.folder)
        # open it in localhost instead
        else:
            fig.show()

    def hist_stim_duration_time(self, df, time_ranges, nbins=0,
                                save_file=True):
        """
        Output distribution of stimulus durations for time ranges.

        Args:
            df (dataframe): dataframe with data from heroku.
            time_ranges (dictionaries): time ranges for analysis.
            nbins (int, optional): number of bins in histogram.
            save_file (bool, optional): flag for saving an html file with plot.
        """
        logger.info('Creating histogram of stimulus durations for time ' +
                    'ranges.')
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
            fig = px.histogram(df[col_dur], nbins=nbins, marginal='rug',
                               color=df['range'], barmode='overlay')
        else:
            fig = px.histogram(df[col_dur], marginal='rug', color=df['range'],
                               barmode='overlay')
        # ticks as numbers
        fig.update_layout(xaxis=dict(tickformat='digits'))
        # update layout
        fig.update_layout(template=self.template)
        # save file
        if save_file:
            self.save_plotly(fig,
                             'hist_stim_duration' +
                             '-'.join(t['start'].strftime('%m.%d.%Y,%H:%M:%S') +  # noqa: E501
                                      '-' +
                                      t['end'].strftime('%m.%d.%Y,%H:%M:%S')
                                      for t in time_ranges),
                             self.folder)
        # open it in localhost instead
        else:
            fig.show()

    def plot_kp(self, df, conf_interval=None, xaxis_title='Time (s)',
                yaxis_title='Percentage of trials with response key pressed',
                xaxis_range=None, yaxis_range=None, save_file=True):
        """Plot keypress data.

        Args:
            df (dataframe): dataframe with keypress data.
            conf_interval (float, optional): show confidence interval defined
                                             by argument.
            xaxis_title (str, optional): title for x axis.
            yaxis_title (str, optional): title for y axis.
            xaxis_range (list, optional): range of x axis in format [min, max].
            yaxis_range (list, optional): range of y axis in format [min, max].
            save_file (bool, optional): flag for saving an html file with plot.
        """
        logger.info('Creating visualisations of keypresses for all data.')
        # calculate times
        times = np.array(range(self.res,
                               df['video_length'].max() + self.res,
                               self.res)) / 1000
        # add all data together. Must be converted to np array to add together
        kp_data = np.array([0.0] * len(times))
        for i, data in enumerate(df['kp']):
            # append zeros to match longest duration
            data = np.pad(data, (0, len(times) - len(data)), 'constant')
            # add data
            kp_data += np.array(data)
        kp_data = (kp_data / i)
        # create figure
        fig = go.Figure()
        # plot keypresses
        fig = px.line(y=kp_data,
                      x=times,
                      title='Keypresses for all stimuli')
        # show confidence interval
        if conf_interval:
            # calculate condidence interval
            (y_lower, y_upper) = self.get_conf_interval_bounds(kp_data,
                                                               conf_interval)
            # plot interval
            fig.add_trace(go.Scatter(name='Upper Bound',
                                     x=times,
                                     y=y_upper,
                                     mode='lines',
                                     fillcolor='rgba(0,100,80,0.2)',
                                     line=dict(color='rgba(255,255,255,0)'),
                                     hoverinfo="skip",
                                     showlegend=False))
            fig.add_trace(go.Scatter(name='Lower Bound',
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
        # save file
        if save_file:
            self.save_plotly(fig, 'kp', self.folder)
        # open it in localhost instead
        else:
            fig.show()

    def plot_kp_video(self, df, stimulus, pp, extention='mp4',
                      conf_interval=None,
                      xaxis_title='Time (s)',
                      yaxis_title='Percentage of trials with ' +
                                  'response key pressed',
                      xaxis_range=None, yaxis_range=None, save_file=True):
        """Plot keypresses with multiple variables as a filter.

        Args:
            df (dataframe): dataframe with keypress data.
            stimulus (str): name of stimulus.
            pp (str, optional): individual participant or 'all'.
            extension (str, optional): extension of stimulus.
            conf_interval (float, optional): show confidence interval defined
                                             by argument.
            xaxis_title (str, optional): title for x axis.
            yaxis_title (str, optional): title for y axis.
            xaxis_range (list, optional): range of x axis in format [min, max].
            yaxis_range (list, optional): range of y axis in format [min, max].
            save_file (bool, optional): flag for saving an html file with plot.
        """
        # todo: implement for 1 pp
        # extract video length
        video_len = df.loc[stimulus]['video_length']
        # calculate times
        times = np.array(range(self.res, video_len + self.res, self.res)) / 1000  # noqa: E501
        # keypress data
        kp_data = df.loc[stimulus]['kp']
        # plot keypresses
        fig = px.line(y=df.loc[stimulus]['kp'],
                      x=times,
                      title='Keypresses for stimulus ' + stimulus)
        # show confidence interval
        if conf_interval:
            # calculate condidence interval
            (y_lower, y_upper) = self.get_conf_interval_bounds(kp_data,
                                                               conf_interval)
            # plot interval
            fig.add_trace(go.Scatter(name='Upper Bound',
                                     x=times,
                                     y=y_upper,
                                     mode='lines',
                                     fillcolor='rgba(0,100,80,0.2)',
                                     line=dict(color='rgba(255,255,255,0)'),
                                     hoverinfo="skip",
                                     showlegend=False))
            fig.add_trace(go.Scatter(name='Lower Bound',
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
        self.kp = fig
        # save file
        if save_file:
            self.save_plotly(fig, 'kp_' + stimulus, self.folder)
        # open it in localhost instead
        else:
            fig.show()

    def plot_kp_video_pp(self, df, dt, stimulus, pp,
                         conf_interval=None, trendline=None,
                         xaxis_title='Time (s)',
                         yaxis_title='response key pressed',
                         xaxis_range=None, yaxis_range=None, save_file=True):
        """Plot keypresses data of one stimulus for 1 participant.

        Args:
            df (dataframe): dataframe with keypress data.
            stimulus (str): name of stimulus.
            pp (str): ID of participant.
            conf_interval (float, optional): show confidence interval defined
                                             by argument.
            xaxis_title (str, optional): title for x axis.
            yaxis_title (str, optional): title for y axis.
            xaxis_range (list, optional): range of x axis in format [min, max].
            yaxis_range (list, optional): range of y axis in format [min, max].
            save_file (bool, optional): flag for saving an html file with plot.
        """
        # todo: implement for 1 pp
        # extract video length
        video_len = df.loc[stimulus]['video_length']
        # calculate times
        times = np.array(range(self.res, video_len + self.res, self.res)) / 1000  # noqa: E501
        # keypress data
        kp_data_time = dt.loc[pp]['video_10-rt-0']
        kp_ar = np.array(kp_data_time)
        kp_data = np.where(kp_ar > 0, 1, 0)
        kp_data_time = kp_ar/100

        # plot keypresses
        fig = px.line(x=kp_data_time,
                      y=kp_data,
                      # animation_frame=kp_data_time,
                      title='Keypresses for stimulus ' + stimulus + ' for participant ' + pp)  # noqa: E501

        # # update layout
        fig.update_layout(template=self.template,
                          xaxis_title=xaxis_title,
                          yaxis_title=yaxis_title,
                          xaxis_range=[0, len(times)],
                          yaxis_range=yaxis_range)
        # save file
        if save_file:
            self.save_plotly(fig, 'kp_' + stimulus, self.folder)
        # open it in localhost instead
        else:
            fig.show()

    def plot_kp_animate(self, df, stimulus, pp='all', extention='mp4',
                        conf_interval=None,
                        xaxis_title='Time (s)',
                        yaxis_title='Percentage of trials with ' +
                                    'response key pressed',
                        xaxis_range=None, yaxis_range=None, save_file=True):
        # extract video length
        video_len = df.loc[stimulus]['video_length']
        # calculate times
        times = np.array(range(self.res, video_len + self.res, self.res)) / 1000  # noqa: E501
        # keypress data
        kp_data = df.loc[stimulus]['kp']
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
                                                                dict(fromcurrent=True,  # noqa: E501
                                                                     transition={'duration': 10},  # noqa: E501
                                                                     frame=dict(redraw=True,  # noqa: E501
                                                                                duration=100))]),  # noqa: E501
                                                     dict(label='Pause',
                                                          method='animate',
                                                          args=[[None],
                                                                dict(fromcurrent=True,  # noqa: E501
                                                                     mode='immediate',  # noqa: E501
                                                                     transition={'duration': 10},  # noqa: E501
                                                                     frame=dict(redraw=True, duration=100))])])])  # noqa: E501
        if conf_interval:
            # calculate condidence interval
            (y_lower, y_upper) = self.get_conf_interval_bounds(kp_data,
                                                               conf_interval)
            # plot interval
            fig.add_trace(go.Scatter(name='Upper Bound',
                                     x=times,
                                     y=y_upper,
                                     mode='lines',
                                     fillcolor='rgba(0,100,80,0.2)',
                                     line=dict(color='rgba(255,255,255,0)'),
                                     hoverinfo='skip',
                                     showlegend=False))
            fig.add_trace(go.Scatter(name='Lower Bound',
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
        # save file
        if save_file:
            self.save_plotly(fig, 'kp_animation' + stimulus, self.folder)
        # open it in localhost instead
        else:
            fig.show()

    def plot_video_data(self, df, stimulus, cols, extention='mp4',
                        conf_interval=None,
                        xaxis_title='Time (s)',
                        yaxis_title='Percentage of trials with ' +
                                    'response key pressed',
                        xaxis_range=None, yaxis_range=None, save_file=True):
        """Plot keypresses with multiple variables as a filter.

        Args:
            df (dataframe): dataframe with keypress data.
            stimulus (str): name of stimulus.
            cols: columns of which to plot
            extention (str, optional): extension of stimulus.
            conf_interval (float, optional): show confidence interval defined
                                             by argument.
            xaxis_title (str, optional): title for x axis.
            yaxis_title (str, optional): title for y axis.
            xaxis_range (list, optional): range of x axis in format [min, max].
            yaxis_range (list, optional): range of y axis in format [min, max].
            save_file (bool, optional): flag for saving an html file with plot.
        """
        # plotly figure to make plots in
        fig = subplots.make_subplots(rows=1,
                                     cols=1,
                                     shared_xaxes=True)
        # plot for all videos
        # buttons = list([dict(label='All',
        #                      method='update',
        #                      args=[{'visible': [True] * df.shape[0]},
        #                            {'title': 'Keypresses for individual stimuli',  # noqa: E501
        #                             'showlegend': True}])])

        for col in cols:
            video_len = df.loc[stimulus]['video_length']
            # calculate times
            times = np.array(range(self.res, video_len + self.res, self.res)) / 1000  # noqa: E501
            # keypress data
            kp_data = df.loc[stimulus][col]
            if type(kp_data) != list:
                kp_data = ast.literal_eval(kp_data)
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
        # save file
        if save_file:
            self.save_plotly(fig, 'video_data_' + stimulus, self.folder)
        # open it in localhost instead
        else:
            fig.show()

    def plot_kp_videos(self, df, xaxis_title='Time (s)',
                       yaxis_title='Percentage of trials with ' +
                                   'response key pressed',
                       xaxis_range=None, yaxis_range=None, save_file=True):
        """Plot keypresses with multiple variables as a filter.

        Args:
            df (dataframe): dataframe with keypress data.
            xaxis_title (str, optional): title for x axis.
            yaxis_title (str, optional): title for y axis.
            xaxis_range (list, optional): range of x axis in format [min, max].
            yaxis_range (list, optional): range of y axis in format [min, max].
            save_file (bool, optional): flag for saving an html file with plot.
        """
        # calculate times
        times = np.array(range(self.res, df['video_length'].max() + self.res, self.res)) / 1000  # noqa: E501
        # plotly
        fig = subplots.make_subplots(rows=1,
                                     cols=1,
                                     shared_xaxes=True)
        # plot for all videos
        for index, row in df.iterrows():
            values = row['kp']
            fig.add_trace(go.Scatter(y=values,
                                     mode='lines',
                                     x=times,
                                     name=os.path.splitext(index)[0]),
                          row=1,
                          col=1)
        buttons = list([dict(label='All',
                             method='update',
                             args=[{'visible': [True] * df.shape[0]},
                                   {'title': 'Keypresses for individual stimuli',  # noqa: E501
                                    'showlegend': True}])])
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
        fig['layout']['title'] = 'Keypresses for individual stimuli'
        # update layout
        fig.update_layout(template=self.template,
                          xaxis_title=xaxis_title,
                          yaxis_title=yaxis_title,
                          xaxis_range=xaxis_range,
                          yaxis_range=yaxis_range)
        # save file
        if save_file:
            self.save_plotly(fig, 'kp_videos', self.folder)
        # open it in localhost instead
        else:
            fig.show()

    def plot_kp_variable(self, df, variable, values=None,
                         xaxis_title='Time (s)',
                         yaxis_title='Percentage of trials with ' +
                                     'response key pressed',
                         xaxis_range=None, yaxis_range=None, save_file=True):
        """Plot figures of values of a certain variable.

        Args:
            df (dataframe): dataframe with keypress data.
            variable (str): variable to plot.
            values (list, optional): values of variable to plot. If None, all
                                     values are plotted.
            xaxis_title (str, optional): title for x axis.
            yaxis_title (str, optional): title for y axis.
            xaxis_range (list, optional): range of x axis in format [min, max].
            yaxis_range (list, optional): range of y axis in format [min, max].
            save_file (bool, optional): flag for saving an html file with plot.
        """
        logger.info('Creating visualisation of keypresses based on values ' +
                    '{} of variable {} .', values, variable)
        # calculate times
        times = np.array(range(self.res, df['video_length'].max() + self.res, self.res)) / 1000  # noqa: E501
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
                data_row = np.pad(data_row, (0, len(times) - len(data_row)),
                                  'constant')
                kp_data = kp_data + data_row
            # divide sums of values over number of rows that qualify
            if df_f.shape[0]:
                kp_data = kp_data / df_f.shape[0]
            extracted_data.append({'value': value,
                                   'data': kp_data})
        # plotly figure
        fig = subplots.make_subplots(rows=1,
                                     cols=1,
                                     shared_xaxes=True)
        # plot each variable in data
        for data in extracted_data:
            fig.add_trace(go.Scatter(y=data['data'],  # noqa: E501
                                     mode='lines',
                                     x=times,
                                     name=str(data['value'])),
                          row=1,
                          col=1)
        # create tabs
        buttons = list([dict(label='All',
                             method='update',
                             args=[{'visible': [True] * len(values)},
                                   {'title': 'All',
                                    'showlegend': True}])])
        # counter for traversing through stimuli
        counter_rows = 0
        for value in values:
            visibility = [[counter_rows == j] for j in range(len(values))]
            visibility = [item for sublist in visibility for item in sublist]  # noqa: E501
            button = dict(label=str(value),
                          method='update',
                          args=[{'visible': visibility},
                                {'title': str(value)}])
            buttons.append(button)
            counter_rows = counter_rows + 1
        # add menu
        updatemenus = [dict(x=-0.15, buttons=buttons, showactive=True)]
        fig['layout']['updatemenus'] = updatemenus
        # update layout
        fig['layout']['title'] = 'Keypresses for ' + variable
        # update layout
        fig.update_layout(template=self.template,
                          xaxis_title=xaxis_title,
                          yaxis_title=yaxis_title,
                          xaxis_range=xaxis_range,
                          yaxis_range=yaxis_range)
        # save file
        if save_file:
            self.save_plotly(fig,
                             'kp_' + variable + '-' +
                             '-'.join(str(val) for val in values),
                             self.folder)
        # open it in localhost instead
        else:
            fig.show()

    def plot_kp_variables_or(self, df, variables, xaxis_title='Time (s)',
                             yaxis_title='Percentage of trials with ' +
                                         'response key pressed',
                             xaxis_range=None, yaxis_range=None,
                             save_file=True):
        """Separate plots of keypresses with multiple variables as a filter.

        Args:
            df (dataframe): dataframe with keypress data.
            variables (list): variables to plot.
            xaxis_title (str, optional): title for x axis.
            yaxis_title (str, optional): title for y axis.
            xaxis_range (list, optional): range of x axis in format [min, max].
            yaxis_range (list, optional): range of y axis in format [min, max].
            save_file (bool, optional): flag for saving an html file with plot.
        """
        logger.info('Creating visualisation of keypresses based on ' +
                    'variables {} with OR filter.', variables)
        # build string with variables
        variables_str = ''
        for variable in variables:
            variables_str = variables_str + '_' + str(variable['variable']) +\
                '-' + str(variable['value'])
        # calculate times
        times = np.array(range(self.res, df['video_length'].max() + self.res, self.res)) / 1000  # noqa: E501
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
            for index, row in df_f.iterrows():  # noqa: E501
                kp_data = kp_data + np.array(row['kp'])
            # divide sums of values over number of rows that qualify
            if df_f.shape[0]:
                kp_data = kp_data / df_f.shape[0]
            extracted_data.append({'value': str(var['variable']) + '-' + str(var['value']),  # noqa: E501
                                   'data': kp_data})
        # plotly figure
        fig = subplots.make_subplots(rows=1,
                                     cols=1,
                                     shared_xaxes=True)
        # plot each variable in data
        for data in extracted_data:
            fig.add_trace(go.Scatter(y=data['data'],  # noqa: E501
                                     mode='lines',
                                     x=times,
                                     name=data['value']),  # noqa: E501
                          row=1,
                          col=1)
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
                                {'title': data['value']}])  # noqa: E501
            buttons.append(button)
            counter_rows = counter_rows + 1
        # add menu
        updatemenus = [dict(x=-0.15, buttons=buttons, showactive=True)]
        fig['layout']['updatemenus'] = updatemenus
        # update layout
        fig['layout']['title'] = 'Keypresses with OR filter'
        # update layout
        fig.update_layout(template=self.template,
                          xaxis_title=xaxis_title,
                          yaxis_title=yaxis_title,
                          xaxis_range=xaxis_range,
                          yaxis_range=yaxis_range)
        # save file
        if save_file:
            self.save_plotly(fig, 'kp_or' + variables_str, self.folder)
        # open it in localhost instead
        else:
            fig.show()

    def plot_kp_variables_and(self, df, plot_names, variables_list,
                              conf_interval=None,
                              xaxis_title='Time (s)',
                              yaxis_title='Percentage of trials with ' +
                                          'response key pressed',
                              xaxis_range=None, yaxis_range=None,
                              save_file=True):
        """Separate plots of keypresses with multiple variables as a filter.

        Args:
            df (dataframe): dataframe with keypress data.
            variables (list): variables to plot.
            conf_interval (float, optional): show confidence interval defined
                                             by argument.
            xaxis_title (str, optional): title for x axis.
            yaxis_title (str, optional): title for y axis.
            xaxis_range (list, optional): range of x axis in format [min, max].
            yaxis_range (list, optional): range of y axis in format [min, max].
            save_file (bool, optional): flag for saving an html file with plot.
        """
        logger.info('Creating visualisation of keypresses based on ' +
                    'variables {} with AND filter.', variables_list)
        # build string with variables
        # create an empty figure, to add scatters to
        fig = subplots.make_subplots(rows=1,
                                     cols=1,
                                     shared_xaxes=True)
        counter = 0
        # retrieve lists to make combined AND plot
        for variables in variables_list:
            variables_str = ''
            for variable in variables:
                variables_str = variables_str + '_' + str(variable['variable'])
            # calculate times
            times = np.array(range(self.res, df['video_length'].max() + self.res, self.res)) / 1000  # noqa: E501
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
            # plot each variable in data
            fig.add_trace(go.Scatter(y=kp_data,  # noqa: E501
                                     mode='lines',
                                     x=times,
                                     name=plot_names[counter]),  # noqa: E501
                          row=1,
                          col=1)

        # show confidence interval
            if conf_interval:
                # calculate condidence interval
                (y_lower, y_upper) = self.get_conf_interval_bounds(kp_data,
                                                                   conf_interval)  # noqa: E501
                # plot interval
                fig.add_trace(go.Scatter(name='Upper Bound',
                                         x=times,
                                         y=y_upper,
                                         mode='lines',
                                         fillcolor='rgba(0,100,80,0.2)',
                                         line=dict(color='rgba(255,255,255,0)'),  # noqa: E501
                                         hoverinfo="skip",
                                         showlegend=False))
                fig.add_trace(go.Scatter(name='Lower Bound',
                                         x=times,
                                         y=y_lower,
                                         fill='tonexty',
                                         fillcolor='rgba(0,100,80,0.2)',
                                         line=dict(color='rgba(255,255,255,0)'),  # noqa: E501
                                         hoverinfo="skip",
                                         showlegend=False))
            # define range of y axis
            if not yaxis_range:
                yaxis_range = [0, max(y_upper) if conf_interval else max(kp_data)]  # noqa: E501
            # update layout
            fig.update_layout(template=self.template,
                              xaxis_title=xaxis_title,
                              yaxis_title=yaxis_title,
                              xaxis_range=xaxis_range,
                              yaxis_range=yaxis_range)
            counter = counter + 1
        # save file
        if save_file:
            self.save_plotly(fig, 'kp_and' + variables_str, self.folder)
        # open it in localhost instead
        else:
            fig.show()

    def map(self, df, color, save_file=True):
        """Map of countries of participation with color based on column in
           dataframe.

        Args:
            df (dataframe): dataframe with keypress data.
            save_file (bool, optional): flag for saving an html file with plot.
        """
        logger.info('Creating visualisation of heatmap of participants by'
                    + ' country with colour defined by {}.', color)
        # create map
        fig = px.choropleth(df,
                            locations='country',
                            color=color,
                            hover_name='country',
                            color_continuous_scale=px.colors.sequential.Plasma)
        # update layout
        fig.update_layout(template=self.template)
        # save file
        if save_file:
            self.save_plotly(fig, 'map_' + color, self.folder)
        # open it in localhost instead
        else:
            fig.show()

    def save_plotly(self, fig, name, output_subdir):
        """
        Helper function to save figure as html file.

        Args:
            fig (plotly figure): figure object.
            name (str): name of html file.
            output_subdir (str): Folder for saving file.
        """
        # build path
        path = tr.settings.output_dir + output_subdir
        if not os.path.exists(path):
            os.makedirs(path)
        # limit name to 255 char
        if len(path) + len(name) > 250:
            name = name[:255 - len(path) - 5]
        file_plot = os.path.join(path + name + '.html')
        # save to file
        py.offline.plot(fig, filename=file_plot)

    def save_fig(self, image, fig, output_subdir, suffix, pad_inches=0):
        """
        Helper function to save figure as file.

        Args:
            image (str): name of figure to save.
            fig (matplotlib figure): figure object.
            output_subdir (str): folder for saving file.
            suffix (str): suffix to add in the end of the filename.
            pad_inches (int, optional): padding.
        """
        # extract name of stimulus after last slash
        file_no_path = image.rsplit('/', 1)[-1]
        # remove extension
        file_no_path = os.path.splitext(file_no_path)[0]
        # turn name into valid file name
        file_no_path = self.slugify(file_no_path)
        # create path
        path = tr.settings.output_dir + output_subdir
        if not os.path.exists(path):
            os.makedirs(path)
        # save file
        plt.savefig(path + file_no_path + suffix,
                    bbox_inches='tight',
                    pad_inches=pad_inches)
        # clear figure from memory
        plt.close(fig)

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
                # The height of the bar is the data value and can be used as
                # the label
                # show demical points
                if decimal:
                    label_text = f'{height:.2f}'
                else:
                    label_text = f'{height:.0f}'
                label_x = x + width / 2
                label_y = y + height / 2
                # plot only when height is greater than specified value
                if height > 0:
                    ax.text(label_x,
                            label_y,
                            label_text,
                            ha='center',
                            va='center')

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
            list of lsits: lower and uppoer bounds.
        """
        # calculate condidence interval
        conf_interval = st.t.interval(conf_interval,
                                      len(data) - 1,
                                      loc=np.mean(data),
                                      scale=st.sem(data))
        # calcuate bounds
        # todo: cross-check if correct
        y_lower = data - conf_interval[0]
        y_upper = data + conf_interval[1]
        return y_lower, y_upper

    def slugify(self, value, allow_unicode=False):
        """
        Taken from https://github.com/django/django/blob/master/django/utils/text.py  # noqa: E501
        Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
        dashes to single dashes. Remove characters that aren't alphanumerics,
        underscores, or hyphens. Convert to lowercase. Also strip leading and
        trailing whitespace, dashes, and underscores.
        """
        value = str(value)
        if allow_unicode:
            value = unicodedata.normalize('NFKC', value)
        else:
            value = unicodedata.normalize('NFKD', value).encode('ascii',
                                                                'ignore').decode('ascii')  # noqa: E501
        value = re.sub(r'[^\w\s-]', '', value.lower())
        return re.sub(r'[-\s]+', '-', value).strip('-_')
