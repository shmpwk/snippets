#!/usr/bin/env python
# -*- coding: utf-8 -*-

import moviepy.editor as mpy
from IPython.display import HTML
from base64 import b64encode
def save_video(frames, path):
    video_path = 'data/video/' + path
    clip = mpy.ImageSequenceClip(frames, fps=30)
    clip.write_videofile(video_path, fps=30)

def play_mp4(path):
    mp4 = open(path, 'rb').read()
    url = "data:video/mp4;base64," + b64encode(mp4).decode()
    return HTML("""<video width=400 controls><source src="%s" type="video/mp4"></video>""" % url)
