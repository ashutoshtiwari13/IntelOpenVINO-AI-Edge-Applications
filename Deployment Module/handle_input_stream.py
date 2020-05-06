'''
Created by @Ashutosh on 7/5
'''

import argparse
import cv2
from inference import Network
from sys import platform

INPUT_STREAM = "test_video.mp4"
# Get correct CPU extension
if platform == "linux" or platform == "linux2":
    CODEC = 0x00000021
    CODEC_COMPATIBLE = 0x7634706d
elif platform == "darwin":
    CODEC = cv2.VideoWriter_fourcc('M','J','P','G')
else:
    print("Unsupported OS.")
    exit(1)

def get_args():
    '''
    Gets the arguments from the command line.
    '''
    parser = argparse.ArgumentParser("Run inference on an input video")
    # -- Create the descriptions for the commands
    i_desc = "Input Video file location"

    # -- Create the arguments
    parser.add_argument("-i", help=i_desc)
    args = parser.parse_args()

    return args
