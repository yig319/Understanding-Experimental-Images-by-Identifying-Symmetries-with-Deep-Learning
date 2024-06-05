import numpy as np

def NormalizeData(data, range=(0,1)):
    return (((data - np.min(data)) * (range[1] - range[0])) / (np.max(data) - np.min(data))) + range[0]