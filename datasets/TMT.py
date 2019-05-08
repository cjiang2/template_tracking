import os
import sys

import numpy as np

def read_tracking_data(filename):
    """
    Read ground truth data in MTF format.
    """
    if not os.path.isfile(filename):
        print("Tracking data file not found:\n ", filename)
        sys.exit()

    data_file = open(filename, 'r')
    data_array = []
    line_id = 0
    for line in data_file:
        if line_id > 0:
            line = line.strip().split()
            coordinate = np.array([[float(line[1]), float(line[3]), float(line[5]), float(line[7])],
                                [float(line[2]), float(line[4]), float(line[6]), float(line[8])]])
            coordinate = np.round(coordinate)
            data_array.append(coordinate.astype(int))
        line_id += 1
    data_file.close()
    return np.array(data_array)
