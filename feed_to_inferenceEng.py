'''
@Created by Ashutosh on 5/4
'''

import argparse
import os
from openvino.inference_engine import IENetwork, IECore

CPU_EXTENSION = "/home/leo/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"

def get_args():
    '''
    Gets the arguments from the command line
    '''
    parser= argparse.ArgumentParser("load an Intermediate Representation into the inference Engine")
    c_desc= "CPU extension fiel location"
    m_desc= "Location of the Model XML file"

    parser.add_argument("-c",help=c_desc, default=CPU_EXTENSION)
    parser.add_argument("-m",help=m_desc)
    args= parser.parse_args()

    return args

def load_to_IE(model_xml , cpu_ext):

    #load the Inference engine API
    plugin=IECore()

    #loaf intermediate representaion files into thier related class
    model_bin = os.path.splitext(model_xml)[0]+ ".bin"
    net= IENetwork(model=model_xml,weights = model_bin)

    plugin.add_extension(cpu_text,"CPU")

    #Get the supported layers of the Network
    supported_layers = plugin.query_network(neywork=net, device_name="CPU")

    #check the supported layer and let teh user know if anything is missing
    unsupported_layers= [l for l in net.layers_keys() if l not in supported_layers]
    if len(unsupported_layers) !=0:
        print("Unsupported layers found:{}".format(unsupported_layers))
        print("Check whether extensions are available to ass to IECore")
        exit(1)

    #Load the network into the Inference engine
    plugin.load_network(net,"CPU")

    print("IR successfully loaded into IE")

    return


def main():
    arg= get_args()
    load_to_IE(args.m , args.c)

if __name__ == "__main__":
    main()


'''
Command :

python feed_to_inferenceEng.py -m /home/workspace/models/human-pose-estimation-0001.xml
'''
