'''
Created by @Ashutosh on 7/5
'''

import argparse
import cv2
from inference import Network
from sys import platform
import numpy as np

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


def capture_stream(args):

    #create a flag for singel images
    image_flag= False
    #chekc if input is webcam
    if args.i== 'CAM':
        args.i=0
    elif args.i.endswith('.jpg') or args.i.endswith('png') or args.i.endswith('bmp'):
        image_flag=True

    #Get and open video capture_stream
    cap= cv2.VideoCapture(args.i)
    cap.open(args.i)

    if not image_flag:
        out= cv2.VideoWriter('output_video.mp4', CODEC_COMPATIBLE,30, (100,100))
    else:
        out=None

    # Process frames until the video ends, or process is exited
    while cap.isOpened():
        # Read the next frame
        flag, frame = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)

        # Re-size the frame to 100x100
        frame = cv2.resize(frame, (100,100))

        ## Add Canny Edge Detection to the frame,
        ##      with min & max values of 100 and 200

        frame = cv2.Canny(frame, 100, 200)
        #To make a 3-channel image
        frame = np.dstack((frame, frame, frame))

        ### Write out the frame, depending on image or video
        if image_flag:
            cv2.imwrite('output_image.jpg', frame)
        else:
            out.write(frame)
        # Break if escape key pressed
        if key_pressed == 27:
            break

    #Close the stream and any windows at the end of the application
    if not image_flag:
        out.release()
    cap.release()
    cv2.destroyAllWindows()

def main():
    args= get_args()
    capture_stream(args)


if __name__ =="__main__":
    main()
