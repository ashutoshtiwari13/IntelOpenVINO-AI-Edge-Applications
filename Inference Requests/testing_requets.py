import argparse
from utils import load_to_IE, preprocessing
from inference import perform_inference
from sys import platform

#CPU extension based on platform
