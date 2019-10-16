import sys
sys.path.append('../')
from jpmesh.jpmesh import *

WD = {
    'input': {
        'extract_raster': '/mnt/7E3B52AF2CE273C0/Thesis/backup_main_folder/Final-Thesis-Dataset/csv_files/'
    },
    'output': {
        'extract_raster': '/mnt/7E3B52AF2CE273C0/Thesis/backup_main_folder/Final-Thesis-Dataset/raster_imgs/org/'
    }
}

ACCIDENT_TYPES = ('0', '{v_to_m}', '{v_to_v_encounter}', '{v_to_v_front}', '{vehicle_own}',
                  '{v_to_v_left}', '{v_to_v_other}', '{v_to_v_rear}', '{v_to_v_right}')

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
    'sns_channel'               : 3,

    'num_factors'               : 4
}

SNS_COMPLAINTS = ['交通', #accident/traffic
                  '台風,増水,雨,竜,浸水,地震', # rain
                  '渋滞,中止,渋滞,遅れ' # congestion
                 ]