'''
Created by @Ashutosh on 6/5
'''
import os
import sys
import logging as log
from openvino.inference_engine import IENetwork,IECore

class Network:
    '''
    Load and store information for the inference_engine and loaded models
    '''

    def __init__(self):
        self.plugin= None
        self.network= None
        self.input_blob= None
        self.output_blob= None
        self.exec_network= None
        self.infer_request = None

    def load_model(self, model, device= "CPU" , cpu_extension=None):
        '''
        Load model by the IR files and Sync requests
        '''
        model_xml=model
        model_bin = os.path.splitext(model_xml)[0] + ".bin"

        #initialize the plugin
        self.plugin =IECore()

        #Adding CPU extension if applicable
        if cpu_extension and "CPU" in device:
            self.plugin.add_extension(cpu_extension,device)

        #Read the IR
        self.network = IENetwork(model= model_xml, weights= model_bin)

        #load the IENetwork into plugin
        self.exec_network = self.plugin.load_network(self.network,device)

        #get the input layer
        self.input_blob = next(iter(self.network.inputs))
        self.output_blob = next(iter(self.network.outputs))

        return

    def get_input_shape(self) :
        '''
        gets the inputs shape of the network
        '''
        return self.network.inputs[self.input_blob].shape

    def  async_inference(self,image):
         '''
         Mkaes a async request , given an input image
         '''
         self.exec_network.start_async(request_id=0,inputs ={self.input_blob :image})
         return


    def wait(self):
        '''
        Checks the status of the inference request
        '''
        status = self.exec_network.requests[0].wait(-1)
        return status


    def extract_output(self):
        return self.exec_network.requests[0].outputs[self.output_blob]
