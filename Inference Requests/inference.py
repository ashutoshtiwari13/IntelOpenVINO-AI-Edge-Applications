import argparse
import cv2
import time
from utils import load_to_IE, preprocessing


CPU_EXTENSION = "/home/leo/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"

def get_args():
    '''
    Gets the arguments from the command line
    '''
    parser= argparse.ArgumentParser("load an Intermediate Representation into the inference Engine")
    c_desc= "CPU extension fiel location"
    m_desc= "Location of the Model XML file"
    i_desc= "The image output location"
    r_desc= "Type of Request , Async (A) or Sync(S)"

    parser.add_argument("-c",help=c_desc, default=CPU_EXTENSION)
    parser.add_argument("-m",help=m_desc)
    parser.add_argument("-i",help=i_desc)
    parser.add_argument("-r",help=r_desc)

    args= parser.parse_args()

    return args

'''
Implementing the Async Inference
'''

def async_inference(exec_net , input_blob, image):

    exec_net.start_async(request_id=0, inputs ={input_blob:image})
    while True:
        status= exec_net.requests[0].wait(-1)
        if status==0:
            break
        else:
            time.sleep(1)
    return exec_net


def sync_inference(exec_net, input_blob, image):

    result= exec_net.infer({input_blob:image})
    return result


def perform_inference(exec_net, request_type,input_image, input_shape):

    '''
    Performs inference on the input image given the Executable Network
    '''

    #get the input input_image
    image= cv2.imread(input_image)
    #extract input_shape
    n,c,h,w=  input_shape

    #call preprocessing from utils
    preprocessed_image= preprocessing(image, h, w)

    input_blob = next(iter(exec_net.inputs))


    '''
    Perform eother Async or Sync inference
    '''
    request_type= request_type.lower()
    if request_type=='a':
        output= async_inference(exec_net, input_blob, preprocessed_image)
    elif request_type=='s':
        output= sync_inference(exec_net, input_blob, preprocessed_image)
    else:
        print("Unknown inference request type , use either 'a' or 's")


    return output

def main():
    args= get_args()
    exec_net, input_shape= load_to_IE(args.m, args.c)
    perform_inference(exec_net,args.r,args.i,input_shape)


if __name__=="__main__":
    main()    
