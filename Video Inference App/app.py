'''
Uses routines already implemented in Inference requets and ../app.py iwth tweaks
'''

import argparse
import cv2
from inference import Network
from sys import platform

INPUT_STREAM = "test_video.mp4"
# Get correct CPU extension
if platform == "linux" or platform == "linux2":
    CPU_EXTENSION = "/home/leo/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"
    CODEC = 0x00000021
    CODEC_COMPATIBLE = 0x7634706d
elif platform == "darwin":
    CPU_EXTENSION = "/home/leo/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension.dylib"
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
    m_desc = "Model XML file location"
    i_desc = "Input file location"
    d_desc = "The device name if not 'CPU'"
    cb_desc= "The color of bounding box to draw ;RED,GREEN , BLUE"
    ct_desc= "Confidence Threshold to use with the bounding boxes"

    # -- Add required and optional groups
    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')

    # -- Create the arguments
    required.add_argument("-m", help=m_desc, required=True)
    optional.add_argument("-i", help=i_desc, default=INPUT_STREAM)
    optional.add_argument("-d", help=d_desc, default='CPU')
    optional.add_argument("-cb",help= cb_desc,default='BLUE')
    optional.add_argument("-ct",help= ct_desc,default=0.6)

    args = parser.parse_args()

    return args

def convert_color(color_string):
    '''
    '''
    colors= {"BLUE" :(255,0,0), "GREEN":(0,255,0),"RED":(0,0,255)}
    out_color= colors.get(color_string)
    if out_color:
        return out_color
    else:
        return colors['BLUE']


def draw_boxes(frame, result, args, width, height):
    '''
    Draw bounding boxes onto the frame.
    '''
    for box in result[0][0]: # Output shape is 1x1x100x7
        conf = box[2]
        if conf >= args.ct:
            xmin = int(box[3] * width)
            ymin = int(box[4] * height)
            xmax = int(box[5] * width)
            ymax = int(box[6] * height)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)
    return frame


def infer_on_video(args):
    #convert the args for color and inference
    args.c= convert_color(args.cb)
    args.ct =float(args.ct)

    ### TODO: Initialize the Inference Engine
    plugin = Network()

    ### TODO: Load the network model into the IE
    plugin.load_model(args.m, args.d, CPU_EXTENSION)
    net_input_shape = plugin.get_input_shape()

    # Get and open video capture
    cap = cv2.VideoCapture(args.i)
    cap.open(args.i)

    # Grab the shape of the input
    width = int(cap.get(3))
    height = int(cap.get(4))

    # Create a video writer for the output video
    # The second argument should be `cv2.VideoWriter_fourcc('M','J','P','G')`
    # on Mac, and `0x00000021` on Linux
    #fourcc= cv2.VideoWriter_fourcc('M','J','P','G')
    out = cv2.VideoWriter('out.mp4', CODEC_COMPATIBLE , 30, (width,height))

    # Process frames until the video ends, or process is exited
    while cap.isOpened():
        # Read the next frame
        flag, frame = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)

        #Pre-process the frame
        p_frame = cv2.resize(frame, (net_input_shape[3], net_input_shape[2]))
        p_frame = p_frame.transpose((2,0,1))
        p_frame = p_frame.reshape(1, *p_frame.shape)

        #Perform inference on the frame
        plugin.async_inference(p_frame)

        # Get the output of inference
        if plugin.wait() == 0:
            result = plugin.extract_output()
            ### TODO: Update the frame to include detected bounding boxes
            frame = draw_boxes(frame, result, args, width, height)
            # Write out the frame
            out.write(frame)

        # Break if escape key pressed
        if key_pressed == 27:
            break

    # Release the out writer, capture, and destroy any OpenCV windows
    out.release()
    cap.release()
    cv2.destroyAllWindows()


def main():
    args = get_args()
    infer_on_video(args)

'''
Command to run :
python app.py -m "/home/workspace/models/frozen_inference_graph.xml" -cb "GREEN" -ct 0.6
'''

if __name__ == "__main__":
    main()
