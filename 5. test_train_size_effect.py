# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 12:03:19 2022

@authors: G. Scabbia, A. Abotaleb, A. Sinopoli

        Qatar Environment and Energy Research Institute (QEERI), 
        Hamad Bin Khalifa University (HBKU), Qatar Foundation, 
        P.O. Box 34110, Doha, Qatar
        
        Corresponding authors: asinopoli@hbku.edu.qa

@project-title: Sulphur Oxidative Coupling of Methane process development 
                and its modelling via Machine Learning
    
@file description: script for testing the training size effect of model 
                      performance and plot the results
                   1. Load the data
                   2. Set the model and testing parameters
                   3. Test the training set size effect on the model performance
                   4. plot the results

    Input data is loaded from ./input data/
    Output figures are saved in ./Figures/


"""

# import libraries
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from keras.layers import Dense
from keras.models import Sequential  
from keras.initializers import GlorotNormal
from keras.regularizers import l2

from sklearn.metrics import mean_squared_error

from pickle import load

def design_ANN(num_of_neurons=3, num_hidden_layers=1, activation_func='sigmoid', 
               num_input=3, num_output=8, opt='adam', 
               loss_function='mean_squared_error', weight_reg=0):

    initializer = GlorotNormal()
    
    # Initialising the ANN
    model = Sequential()

    # Adding the input layer and the first hidden layer
    model.add(Dense(units = num_of_neurons, 
                    activation = activation_func, 
                    kernel_initializer=initializer,
                    kernel_regularizer=l2(weight_reg), 
                    input_dim = num_input))

    if num_hidden_layers > 1:
        for layer in range(1, num_hidden_layers):
            # Adding the additional hidden layer
            model.add(Dense(units = num_of_neurons, 
                            activation = activation_func, 
                            kernel_initializer=initializer))

    # Adding the output layer
    model.add(Dense(units = num_output, 
                    activation = 'linear', 
                    kernel_initializer=initializer))
    
    # Compiling the ANN
    model.compile(optimizer = opt, loss = loss_function)
    
    return model


def main():
    
    input_var = ['temperature', 'pressure', 'mass_ratio']
    
    output_var = ['H2S_conv', 'CS2_conv', 'naphthalene', 'p_Xylene', 
                  'toluene', 'benzene', 'H2S', 'methane']
    
    ## Load the data
    
    X_train = pd.read_csv('./input data/X_train.csv')
    
    y_train = pd.read_csv('./input data/y_train.csv')
    
    X_val = pd.read_csv('./input data/X_val.csv')
    y_val = pd.read_csv('./input data/y_val.csv')
    
    X_test = pd.read_csv('./input data/X_test.csv')
    y_test = pd.read_csv('./input data/y_test.csv')
    
    # load scaling functions
    sc_input = load(open('./input data/sc_input.pkl', 'rb'))
    sc_output = load(open('./input data/sc_output.pkl', 'rb'))
    
    print('Train set size: ' +str(X_train.shape[0]))
    print('Validation set size: ' +str(X_val.shape[0]))
    print('Test set size: ' +str(X_test.shape[0]))
    
    ## Model settings and parameters lists
    
    # Model hyperparameters
    
    # fixed settings
    num_input = len(input_var)
    num_output = len(output_var)
    numb_of_epochs = 1000
    loss_function = 'mean_squared_error'
    
    # hyperparameters:
    num_hidden_layers = 2
    num_of_neurons = 5
    activation_func = 'tanh'
    weight_reg = 0
    opt = 'Adamax'
    
    training_size_DIM = [100, 250, 500, 1000, 2500, 5000, 10000, 20000]
    plot_sizes = [100, 250, 500, 1000]
    
    ## Test training size effect of model performance and plot the results
    
    # limits for the result plot of each variable
    limits = [
        [-0.1, 1.1],
        [-0.1, 1.1],
        [-0.1, 3.3],
        [-0.1, 1.1],
        [-0.1, 0.66],
        [-0.1, 1.1],
        [-0.1, 38],
        [27, 110]]
    
    
    history_result = []
    eval_result = pd.DataFrame(columns = ['train_size', 'train_acc', 'val_acc', 'test_acc'])
    output_mse_result = pd.DataFrame(columns = ['train_size']+output_var)
    
    X_train_base = pd.DataFrame(sc_input.inverse_transform(X_train), 
                                columns= X_train.columns, index=X_train.index)
    y_train_base = pd.DataFrame(sc_output.inverse_transform(y_train), 
                                columns= y_train.columns, index=y_train.index)
            
    X_test_base = pd.DataFrame(sc_input.inverse_transform(X_test), 
                               columns= X_test.columns, index=X_test.index)
    y_test_base = pd.DataFrame(sc_output.inverse_transform(y_test), 
                               columns= y_test.columns, index=y_test.index)
    
    X_data = X_train.append(X_test, ignore_index=False)
    
    X_data_base = X_train_base.append(X_test_base, ignore_index=False)
    
    
    ### plot test points for each output
    n = 0
    MR = X_test_base.mass_ratio.unique()[-1] #0.42
    var = 'pressure'
    x_label = 'temperature'
    x_label_str = 'temperature (°C)'
    sel_value = 1.0
    i , j = 0 , 0    
    
    colors = sns.color_palette()
    
    test_data = y_test_base.copy(deep=True)
    
    for variable in input_var:
        test_data[variable] = X_test_base[variable]
    
    fig, ax = plt.subplots(4,2, figsize=(12,15))
    
    c = 0
    
    for col in output_var:
    
        test = test_data.loc[(test_data['mass_ratio'] == MR)&(test_data[var] == sel_value)]
        
    
        ax[i,j].scatter(test[x_label], 
                             test[col], color='black', marker='x')
        
        ax[i,j].set_xlabel(x_label_str)
        ax[i,j].set_ylabel('Comparison')
        
        ax[i,j].set_title(col)
        
        ax[i,j].set_ylim(limits[c])
        
        c = c + 1
        
        j = j + 1
        if j == 2:
            j = 0
            i = i + 1
    
    for size in training_size_DIM:
        
        X_train_sample = X_train.sample(n=size, random_state=1)
        
        y_train_sample = y_train.loc[X_train_sample.index]
        
        print('Testing: train set size ' +str(X_train_sample.shape[0]))
        
        # Fitting the ANN to the Training set
        
        model = design_ANN(num_of_neurons, num_hidden_layers, activation_func, 
                   num_input, num_output, opt, loss_function, weight_reg)
        
        #model.summary()
    
        history = model.fit(X_train_sample, y_train_sample, 
                            batch_size = 32, 
                            epochs = numb_of_epochs, 
                            verbose=0, 
                            validation_data=(X_val, y_val))
        
        history_result.append(history.history)
        
        # model evaluation
        
        train_acc = model.evaluate(X_train, y_train, verbose=0)
        val_acc = model.evaluate(X_val, y_val, verbose=0)
        test_acc = model.evaluate(X_test, y_test, verbose=0)
    
        print('   Train: %.5f, Validation: %.5f, Test: %.5f' % (train_acc, val_acc, test_acc))
        
        eval_result.loc[eval_result.shape[0]] = [size, train_acc, val_acc, test_acc]
            
        # evaluate single-output error
        
        pred = pd.DataFrame(model.predict(X_test), columns=y_val.columns)
        pred.index = y_val.index.values
    
        output_mse = [size]
        for col in pred.columns:
            
            output_mse.append(mean_squared_error(y_test[col], pred[col]))
        
            #print(col+' MSE: '+str(round(mse, 5)))
    
        #print('   Final error: '+str(round(tot_mse, 5)))
        
        output_mse_result.loc[output_mse_result.shape[0]] = output_mse
        
        if size in plot_sizes:
        
            # plot resulting prediction for each output
    
            pred = pd.DataFrame(sc_output.inverse_transform(model.predict(X_data)), 
                                columns= y_test.columns, index=X_data.index)
    
            for variable in input_var:
                pred[variable] = X_data_base[variable]
    
            prediction = pred.loc[(pred['mass_ratio'] == MR)&(pred[var] == sel_value)]
    
            i , j = 0 , 0
    
            for col in output_var:
    
                ax[i,j].plot(prediction.sort_values(by=x_label)[x_label], 
                             prediction.sort_values(by=x_label)[col], color=colors[n])
    
                j = j + 1
                if j == 2:
                    j = 0
                    i = i + 1
        
            n = n + 1
    
    ax_leg = fig.add_subplot(111)
    ax_leg.axis('off')
    ax_leg.plot(np.NaN, np.NaN, 'x', color='k', label='test obs.')
    n=0
    for size in plot_sizes: 
        ax_leg.plot(np.NaN, np.NaN, 's', color=colors[n], label=str(size))
        n=n+1
    ax_leg.legend(ncol=len(training_size_DIM)+1, bbox_to_anchor=(0.5, -0.105), 
                  loc='lower center', title='Training set size')
    
    fig.tight_layout()
    plt.savefig('./Figures/trainSize_comparison_outputs.svg')
    plt.savefig('./Figures/trainSize_comparison_outputs.png')
    
    
    
if __name__ == "__main__":
    main()
    
    
  