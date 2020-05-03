'''
Created by Ashutosh ,on 5/2
'''

import cv2
import numpy as np


def pose_selection(output, input_shape):
    '''
    Handles the output of the Pose Estimation model.
    Returns ONLY the keypoint heatmaps
    '''
    #Extract only the second blob output (keypoint heatmaps)
    heatmaps = output['Mconv7_stage2_L2']
    #Resize the heatmap back to the size of the input
    # Create an empty array to handle the output map
    out_heatmap = np.zeros([heatmaps.shape[1], input_shape[0], input_shape[1]])
    # Iterate through and re-size each heatmap
    for h in range(len(heatmaps[0])):
        out_heatmap[h] = cv2.resize(heatmaps[0][h], input_shape[0:2][::-1])

    return out_heatmap


def text_selection(output, input_shape):
    '''
    Handles the output of the Text Detection model.
    Returns the text/no text classification of each pixel
    '''
     #Extract only the first blob output (text/no text classification)
    text_classes = output['model/segm_logits/add']
    #Resize this output back to the size of the input
    out_text = np.empty([text_classes.shape[1], input_shape[0], input_shape[1]])
    for t in range(len(text_classes[0])):
        out_text[t] = cv2.resize(text_classes[0][t], input_shape[0:2][::-1])

    return out_text


def car_type_color_selection(output, input_shape):
    '''
    Handles the output of the Car Metadata model.
    Returns two integers: the argmax of each softmax output.
    The first is for color, and the second for type.
    '''
    # Get rid of unnecessary dimensions
    color = output['color'].flatten()
    car_type = output['type'].flatten()
    #Get the argmax of the "color" output
    color_pred = np.argmax(color)
    #Get the argmax of the "type" output
    type_pred = np.argmax(car_type)

    return color_pred, type_pred


def output_of_selection(model_type):
    '''
    Returns the related function to handle an output,
        based on the model_type being used.
    '''
    if model_type == "POSE":
        return pose_selection
    elif model_type == "TEXT":
        return text_selection
    elif model_type == "CAR_TYPE_COLOR":
        return car_type_color_selection
    else:
        return None

'''
Preprocessing step - See preprocess_inputs.ipynb
'''

def preprocessing(input_image, height, width):
    '''
    Given an input image, height and width:
    - Resize to width and height
    - Transpose the final "channel" dimension to be first
    - Reshape the image to add a "batch" of 1 at the start
    '''
    image = np.copy(input_image)
    image = cv2.resize(image, (width, height))
    image = image.transpose((2,0,1))
    image = image.reshape(1, 3, height, width)

    return image


'''
Arguments for Testing the code
python app.py -i "images/blue-car.jpg"
     -t "CAR_META"
     -m "/home/workspace/models/vehicle-attributes-recognition-barrier-0039.xml"
     -c "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"
'''
