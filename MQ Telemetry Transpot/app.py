'''
Uses routines already implemented in Inference requets and ../app.py with tweaks
'''

import argparse
import cv2
from inference import Network
import sys
from sys import platform
import numpy as np
import socket
import json
from random import randint

#import mqtt libraries
import paho.mqtt.client as mqtt


INPUT_STREAM = "test_video.mp4"
ADAS_MODEL = "/home/workspace/models/semantic-segmentation-adas-0001.xml"
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

CLASSES = ['road','sidewalk','building','wall','fence','pole','traffic_light','traffic_sign','vegetation','terrain','sky','person','rider','car','truck','bus','train','motorcycle','bicycle','ego-vehicle']

#MQTT server env variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST= IPADDRESS
MQTT_PORT= 3002       #3001 could be used
MQTT_KEEPALIVE_INTERVAL = 60


def get_args():
    '''
    Gets the arguments from the command line.
    '''
    parser = argparse.ArgumentParser("Run inference on an input video")
    # -- Create the descriptions for the commands
    i_desc = "Input file location"
    d_desc = "The device name if not 'CPU'"

    # -- Create the arguments
    parser.add_argument("-i", help=i_desc, default=INPUT_STREAM)
    parser.add_argument("-d", help=d_desc, default='CPU')
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


# def draw_boxes(frame, result, args, width, height):
#     '''
#     Draw bounding boxes onto the frame.
#     '''
#     for box in result[0][0]: # Output shape is 1x1x100x7
#         conf = box[2]
#         if conf >= args.ct:
#             xmin = int(box[3] * width)
#             ymin = int(box[4] * height)
#             xmax = int(box[5] * width)
#             ymax = int(box[6] * height)
#             cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)
#     return frame

def draw_masks(result,width,height):
    '''
    Drawa semantic masks of the classes on to the frame
    '''
    classes =cv2.resize(result[0].transpose((1,2,0)),(width,height), interpolation =cv2.INTER_NEAREST)
    unique_classes =np.unique(classes)
    out_mask = classes * (255/20)

    #stack the mask so ffmpeg understans it
    out_mask = np.dstack((out_mask,out_mask,out_mask))
    out_mask = np.uint8(out_mask)

    return out_mask ,unique_classes


def get_class_names(class_nums):
    class_names= []
    for i in class_nums:
        class_names.append(CLASSES[int(i)])
    return class_names



def infer_on_video(args,model):

    # Connect to the MQTT server
    client= mqtt.Client()
    client.connect(MQTT_HOST,MQTT_PORT,MQTT_KEEPALIVE_INTERVAL)


    #Initialize the Inference Engine
    plugin = Network()

    ## Load the network model into the IE
    plugin.load_model(model, args.d, CPU_EXTENSION)
    net_input_shape = plugin.get_input_shape()

    # Get and open video capture
    cap = cv2.VideoCapture(args.i)
    cap.open(args.i)

    # Grab the shape of the input
    width = int(cap.get(3))
    height = int(cap.get(4))

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
            ## Draw the output mask onto the input
            out_frame, classes = draw_masks(result,width,height)
            class_names =get_class_names(classes)
            speed =randint(50,70)

            #Sedning the class  names , speed details to the mqtt server
            client.publish("class",json.dumps({"class_names":class_names}))
            client.publish("seedometer",json.dumps({"speed" :speed}))



        #send frame to the concerned server ,say, ffmpeg server here
        sys.stdout.buffer.write(out_frame)
        sys.stdout.flush()

        # Break if escape key pressed
        if key_pressed == 27:
            break

    # Release the capture, and destroy any OpenCV windows and disconnect form MQTT
    cap.release()
    cv2.destroyAllWindows()
    client.disconnect()

def main():
    args = get_args()
    model =ADAS_MODEL
    infer_on_video(args,model)

if __name__ == "__main__":
    main()
