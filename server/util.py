import json
import pickle
import os
import numpy as np
__locations = None
__data_columns = None
__model = None

def get_estimated_price(location,sqft,bhk,bath):
    try:
        loc_index = __data_columns.index(location.lower())
    except:
        loc_index = -1
    x_input=np.zeros(len(__data_columns))
    x_input[0]=sqft
    x_input[1]=bath
    x_input[2]=bhk
    if loc_index >= 0:
        x_input[loc_index] = 1
    return round(__model.predict([x_input])[0],2)

def get_location_names():
    return __locations

def load_saved_artifacts():
    print("loading saved artifacts...start")
    global __data_columns
    global __locations
    global __model

    
    artifact_path = os.path.join(os.path.dirname(__file__), 'artifact')

    
    with open(os.path.join(artifact_path, "columns.json"), 'r') as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[3:]

    
    with open(os.path.join(artifact_path, "banglore_home_prices_model.pickle"), 'rb') as f:
        __model = pickle.load(f)

    print("loading saved artifacts...done")

if __name__ == '__main__':
    load_saved_artifacts()
    print(get_location_names())
    
