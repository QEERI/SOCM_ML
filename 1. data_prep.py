# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 10:21:22 2022

@authors: G. Scabbia, A. Abotaleb, A. Sinopoli

        Qatar Environment and Energy Research Institute (QEERI), 
        Hamad Bin Khalifa University (HBKU), Qatar Foundation, 
        P.O. Box 34110, Doha, Qatar
        
        Corresponding authors: asinopoli@hbku.edu.qa

@project-title: Sulphur Oxidative Coupling of Methane process development 
                and its modelling via Machine Learning
    
@file description: script for preparing the data for further processing and 
                   modelling.
                   1. Load the raw data
                   2. Scale each variable to [0,1] interval
                   3. Split the data in Train (80%), Validation (10%), 
                        and Test (10%) sets. Save the split sets.

    Raw input data is loaded from ./input data/
    Output processed data are stored in ./input data/


"""

# import libraries
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from pickle import dump
from sklearn.model_selection import train_test_split

# global vars
RAND_INT = 321           # Controls the shuffling applied to the data before 
                         #    applying the split. Pass an int for reproducible 
                         #    output across multiple function calls.

VALTEST_SET_SIZE = 0.2   # portion of the dataset assigned to the test and 
                         #    validation set together
TEST_SET_SIZE = 0.5      # portion of the val_test set assigned to the sole test set. 
                         #    The remaining is the resulting validation set

def main():
    
    input_var = ['temperature', 'pressure', 'mass_ratio']
    
    output_var = ['H2S_conv', 'CS2_conv', 'naphthalene', 'p_Xylene', 'toluene', 
                  'benzene', 'H2S', 'methane']
    
    ## load the raw data    
    data = pd.read_csv('./input data/input_data.csv')
    
    ## Scaling the data: MaxMin normalization of input and outputs
    
    sc_input = MinMaxScaler()
    sc_output = MinMaxScaler()
    
    X_data_sc = pd.DataFrame(sc_input.fit_transform(data[input_var]), 
                             columns=data[input_var].columns)
    
    y_data_sc = pd.DataFrame(sc_output.fit_transform(data[output_var]), 
                             columns=data[output_var].columns)
    
    # save the scaler functions
    dump(sc_input, open('./input data/sc_input.pkl', 'wb'))
    dump(sc_output, open('./input data/sc_output.pkl', 'wb'))
    
    ## Train/validation/test split
    
    X_train, X_Valtest, y_train, y_Valtest = train_test_split(
        X_data_sc, y_data_sc, test_size = VALTEST_SET_SIZE, random_state = RAND_INT)
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_Valtest, y_Valtest, test_size = TEST_SET_SIZE, random_state = RAND_INT)
    
    X_train.to_csv('./input data/X_train.csv', index=True, index_label='index_data')
    y_train.to_csv('./input data/y_train.csv', index=True, index_label='index_data')
    
    X_val.to_csv('./input data/X_val.csv', index=True, index_label='index_data')
    y_val.to_csv('./input data/y_val.csv', index=True, index_label='index_data')
    
    X_test.to_csv('./input data/X_test.csv', index=True, index_label='index_data')
    y_test.to_csv('./input data/y_test.csv', index=True, index_label='index_data')
    

if __name__ == "__main__":
    main()