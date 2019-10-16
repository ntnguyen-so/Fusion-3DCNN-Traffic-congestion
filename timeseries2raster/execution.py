## Importing the libraries
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from config import *
from PIL import Image
import sys
sys.path.append('../')
from data_utils.DataParser import *
from data_utils.DataReader import *
from data_utils.DataScaler import *
from jpmesh.jpmesh import *

def parse_data(dp, data, col_conf, target_col):
    # datetime
    data = dp.removeDelimiters(data, col_conf.index('datetime'), ('-', ' ', ':', '+09'))    
    data = dp.convertInt(data, col_conf.index('datetime'))

    # list factors    
    data = dp.countElements(data, col_conf.index('numsegments'), ',')

    # integer factors
    data = dp.convertInt(data, col_conf.index('meshcode'))
    data = dp.convertInt(data, col_conf.index('rainfall'))
    data = dp.convertInt(data, col_conf.index('congestion'))

    # BoW SNS
    data = dp.removeDelimiters(data, col_conf.index('sns'), ('{', '}'))
    data = dp.convertBoWType(data, col_conf.index('sns'), SNS_COMPLAINTS)

    # meshcode to coordination
    data = dp.removeDelimiters(data, col_conf.index('center'), ('POINT(', ')'))    
    data = dp.convertLocation(data, col_conf.index('center'), ' ')
    
    return data

def convert_idx2time(id, step=5):
    hh = mm = ss = 0
    step_per_min = int(60/step)

    if id < step_per_min:
        mm = id * step
    else:
        hh = id // step_per_min
        mm = id % step_per_min * 5
    
    time = '{:02d}{:02d}'.format(hh, mm)
    return time

def dump_factor(path, factor_map):
    np.savez_compressed(path, factor_map)

def generate_factor_map(path, data, col_conf, factor_config):
    loc = {}

    date = data[0, col_conf.index('datetime')]
    date //= 1000000

    timecode = 0
    j = 0

    timecodesPerDay = 288

    while timecode < timecodesPerDay:
        time = convert_idx2time(timecode)
        starting_time = str(date) + time
        factor_map = np.zeros((factor_config['width'], factor_config['height'],factor_config['num_factors']))
        
        while j < data.shape[0]:
            ending_time = data[j, col_conf.index('datetime')] // 100
            if str(ending_time) > starting_time:
                break

            # get center coordination of each mesh
            meshcode = data[j, col_conf.index('meshcode')]
            mesh = parse_mesh_code(str(meshcode))
            mesh_center = mesh.south_west + (mesh.size / 2.0)
            latitude = mesh_center.lat.degree
            longitude = mesh_center.lon.degree

            # calculate relative positive on raster image
            # a side note: in our orignal dataset, there are coordinations of locations ('center' column)
            # if this information is used, uncomment the next 2 lines to replace the latters
            # otherwise just let as it is to use the current implementation of meshcode
            #loc['x'] = int((data[j, col_conf.index('latitude')] - factor_config['start_lat']) // factor_config['offset_lat'])
            #loc['y'] = int((data[j, col_conf.index('longitude')] - factor_config['start_long']) // factor_config['offset_long'])
            loc['x'] = int((latitude - factor_config['start_lat']) // factor_config['offset_lat'])
            loc['y'] = int((longitude - factor_config['start_long']) // factor_config['offset_long'])

            if loc['x'] >= RASTER_CONFIG['width'] or loc['y'] >= RASTER_CONFIG['height']:
                j += 1
                continue
            
            # assign sensing data to corresponding location on raster image
            congestion = data[j, col_conf.index('congestion')] * data[j, col_conf.index('numsegments')]
            rainfall = data[j, col_conf.index('rainfall')]
            accident = data[j, col_conf.index('accident')]  
            if isinstance(accident, str) == True:
                accident = 1          
            if accident > 0:
                accident = 1

            sns      = data[j, col_conf.index('sns')]
            if sns > 0:
                sns = 1

            factor_map[loc['x'] ,loc['y'],factor_config['congestion_channel']]  = congestion
            factor_map[loc['x'] ,loc['y'],factor_config['rainfall_channel']]    = rainfall
            factor_map[loc['x'] ,loc['y'],factor_config['accident_channel']]    = accident
            factor_map[loc['x'] ,loc['y'],factor_config['sns_channel']]         = sns

            j += 1

        print('Generating raster image for', starting_time)
        factor_map = np.flip(factor_map, axis=0)
        dump_factor(WD['output']['extract_raster'] + starting_time, factor_map)
        del factor_map

        timecode += 6

if __name__ == "__main__":
    # ============================================ #
    # Needed columns for extraction
    col_name = ['datetime', 'meshcode', 'rainfall', 'numsegments', 'congestion', 'accident', 'sns', 'center']    
    col_idx  = [1         , 2         , 5         , 6            , 9           , 15        , 13,    17   ]
    target_col = len(col_name) - 1

    # ============================================ #
    # Data location
    data_path = WD['input']['extract_raster']           # path to time series dataset
    output_raster_path = WD['output']['extract_raster'] # path to raster image storage

    # ============================================ #
    # Read data
    data_files = os.listdir(data_path)
    data_files.sort()
    for i in range(len(data_files)):
        data_files[i] = data_path + data_files[i]
    
    dr = DataReader(data_files, col_idx)
    ds = DataScaler()
    dp = DataParser()

    needed_col_name = col_name[:-1]
    needed_col_name.append('latitude')
    needed_col_name.append('longitude')

    if len(sys.argv) > 2:
        start = int(sys.argv[1])
        end = int(sys.argv[2])
    else:
        start = data_files.index('/mnt/7E3B52AF2CE273C0/Thesis/backup_main_folder/Final-Thesis-Dataset/csv_files/20140916.csv')
        end = data_files.index('/mnt/7E3B52AF2CE273C0/Thesis/backup_main_folder/Final-Thesis-Dataset/csv_files/20141031.csv')

    for file_id in range(start, end):
        #if '2015' not in data_files[file_id]:
        #    continue
        dr_tmp = DataReader([data_files[file_id]], col_idx)
        dr_tmp.read(delimiter='\t')
        data = dr_tmp.getData()
        data = parse_data(dp, data, col_name, target_col)
        
        generate_factor_map(output_raster_path, data, needed_col_name, RASTER_CONFIG)
        del data
        del dr_tmp
    