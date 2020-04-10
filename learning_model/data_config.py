FACTOR = {
    # factor channel index
    'Input_congestion'        : 0,
    'Input_rainfall'          : 1,
    'Input_accident'          : 2,
    'Input_sns'               : 3,   
    'default'                 : 0
}

MAX_FACTOR = {
    'Input_congestion'        : 2600,
    'Input_rainfall'          : 131,
    'Input_sns'               : 1,
    'Input_accident'          : 1,
    'default'                 : 2600,
}

LINK_FACTOR = {
    'Input_congestion'      : (4,5,6,7),
    'Input_rainfall'        : (2,3,6,7),
    'Input_accident'        : (1,3,5,7)
}

BOUNDARY_AREA = {
    0 : [ 20, 80,   50,  100],
    1 : [ 40, 100,  100, 180],
    2 : [ 20, 80,   180, 250]
}

PADDING = {
    0 : [ 0,  60, 30, 80],
    1 : [ 0,  60,  0, 80],
    2 : [ 0,  60,  0, 70]
}

GLOBAL_SIZE_X = [6, 60, 80, 4]
GLOBAL_SIZE_Y = [6, 60, 80, 1]
