import sys
sys.path.append('../')
from jpmesh.jpmesh import QuarterMesh, Coordinate, Angle
import math
import numpy as np

WD = {
    'input': {        
        'sequence_prepdataset': '/mnt/7E3B52AF2CE273C0/Thesis/backup_main_folder/Final-Thesis-Dataset/raster_imgs/sns_2channels/'    # path to multimodal raster images
    },
    'output': {
        'sequence_prepdataset': '/mnt/7E3B52AF2CE273C0/Thesis/backup_main_folder/Final-Thesis-Dataset/seq_raster_imgs/short/' # path to 3D raster images
    }
}

RASTER_CONFIG = {
    # factor map
    'width'             : 1060,
    'height'            : 1060,
    'offset_lat'        : 0.002083333,
    'offset_long'       : 0.003125,
    'start_lat'         : 33.584375,
    'start_long'        : 134.0015625,
    'pixel_area'        : 0.25 ** 2, # 0.25km
    
    # factor channel index
    'congestion_channel'        : 0,
    'rainfall_channel'          : 1,
    'accident_channel'          : 2,
    'sns_traffic'               : 3,
    'sns_environment'           : 4,

    'num_factors'               : 5
}

SEQUENCE = {
    'crop' : {
        'xS'    :   560,
        'xE'    :   660,
        'yS'    :   200,
        'yE'    :   450
    },

    'inp_len'    :   6, # 6 steps 
    'inp_delta'  :   1, # 12*5mins=1h
    'out_len'    :   3, # 3 steps
    'out_delta'  :   1, # 12*5mins=3h
    'out_factor' : RASTER_CONFIG['congestion_channel']
}
