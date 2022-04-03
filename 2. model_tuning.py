# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 10:49:43 2022

@authors: G. Scabbia, A. Abotaleb, A. Sinopoli

        Qatar Environment and Energy Research Institute (QEERI), 
        Hamad Bin Khalifa University (HBKU), Qatar Foundation, 
        P.O. Box 34110, Doha, Qatar
        
        Corresponding authors: asinopoli@hbku.edu.qa

@project-title: Sulphur Oxidative Coupling of Methane process development 
                and its modelling via Machine Learning
    
@file description: script for tuning the hyperparameter of the ANN.
                   1. Load the split data
                   2. Tune the Activation function
                   3. Tune the ANN architecture
                   4. Tune the Optimization function
                   5. Tune the Layer weight regularizers

    Input data is loaded from ./input data/
    validation results are stored in:
        activation function:    ./validation_results/activation_function/
        architecture:           ./validation_results/architecture/
        optimizaer:             ./validation_results/opt/
        regularization term:    ./validation_results/L2_reg/


"""

# import libraries
import pandas as pd

from pickle import load

from keras.initializers import GlorotNormal
from keras.layers import Dense
from keras.models import Sequential
from keras.regularizers import l2

from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import PReLU



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
    
    ## Load the data from the train/validation/test split
    
    X_train = pd.read_csv('./input data/X_train.csv')
    
    y_train = pd.read_csv('./input data/y_train.csv')
    
    X_val = pd.read_csv('./input data/X_val.csv')
    y_val = pd.read_csv('./input data/y_val.csv')

    print('Train set size: ' +str(X_train.shape[0]))
    print('Validation set size: ' +str(X_val.shape[0]))
    
    ## Hyperparameter tuning Validation
    
    # fixed settings
    num_input = len(input_var)
    num_output = len(output_var)    
    loss_function = 'mean_squared_error'
    
    # hyperparameters:
    num_hidden_layers = [1, 2, 3]
    num_of_neurons = [5, 6, 7]
    activation_func = ['sigmoid', 'tanh', 'relu', 'LeakyReLU', 'PReLU', 'elu']    
    optimizers = ['SGD', 'RMSprop', 'Adam', 'Adadelta', 'Adagrad', 'Adamax', 
                  'Nadam', 'Ftrl']
    weight_reg = [0, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]

    
    # 1. Tune the Activation function
    
    # hyperparameters setting
    neurons = 6
    layers = 1
    opt = 'adam'
    numb_of_epochs = 300
    weight_reg = 0
    
    # store the evaluation results in eval_result
    eval_result = pd.DataFrame(columns = ['function', 'train_acc', 'val_acc'])
    
    for function in activation_func:
        
        if function == 'LeakyReLU':
            act_func = LeakyReLU(alpha=0.01)
        elif function == 'PReLU':
            act_func = PReLU()  
        else : 
            act_func = function
            
             
        print('Validating: '+str(function))
        
        # design the model
        model = design_ANN(neurons, layers, act_func, 
                   num_input, num_output, opt, loss_function, weight_reg)
        
        # train the model
        history = model.fit(X_train, y_train, 
                            batch_size = 32, 
                            epochs = numb_of_epochs, verbose=1, 
                            validation_data=(X_val, y_val))
        
        # convert the history.history dict to a pandas DataFrame and save it for later use    
        hist_df = pd.DataFrame(history.history)
        
        hist_csv_file = './validation_results/activation_function/history_'+str(function)+'.csv'
        with open(hist_csv_file, mode='w') as file:
            hist_df.to_csv(file)
        
        # Evaluate the model
        
        train_acc = model.evaluate(X_train, y_train, verbose=0)
        val_acc = model.evaluate(X_val, y_val, verbose=0)    
        
        print('\ttrain acc: %.3f \tval acc: %.3f' % ( train_acc, val_acc))
        
        eval_result.loc[eval_result.shape[0]] = [str(function), train_acc, val_acc]
        
    # save the result
    with open('./validation_results/activation_function/result_models.csv', mode='w') as file:
            eval_result.to_csv(file)
    
    # 2. Tune the ANN architecture
    
    # hyperparameters setting
    act_func = 'tanh'
    opt = 'adam'
    numb_of_epochs = 2000
    weight_reg = 0
    
    # store the evaluation results in eval_result
    eval_result = pd.DataFrame(columns = ['num_hidden_layers', 'num_nodes', 
                                          'train_acc', 'val_acc'])
    
    for layers in num_hidden_layers:
        
        for nodes in num_of_neurons:
            
            architecture = str(layers)+'_'+str(nodes)
            
            print('Validating: '+str(layers) +'hidden layers, '+str(nodes)+' hidden nodes')
    
            # design the model
            model = design_ANN(nodes, layers, act_func, 
                       num_input, num_output, opt, 
                       loss_function, weight_reg)
            
            model.summary()
    
            # train the model
            history = model.fit(X_train, y_train, 
                                batch_size = 32, 
                                epochs = numb_of_epochs, 
                                verbose=1, 
                                validation_data=(X_val, y_val))
    
            # convert the history.history dict to a pandas DataFrame and save it for later use    
            hist_df = pd.DataFrame(history.history)
    
            hist_csv_file = './validation_results/architecture/history_'+architecture+'.csv'
            with open(hist_csv_file, mode='w') as file:
                hist_df.to_csv(file)
    
            # Evaluate the model
    
            train_acc = model.evaluate(X_train, y_train, verbose=0)
            val_acc = model.evaluate(X_val, y_val, verbose=0)    
            print('\ttrain acc: %.3f \tval acc: %.3f' % ( train_acc, val_acc))
    
            eval_result.loc[eval_result.shape[0]] = [layers, nodes, train_acc, val_acc]
    
    # save the result
    with open('./validation_results/architecture/result_models.csv', mode='w') as file:
            eval_result.to_csv(file)
            
    # 3. Tune the Optimization function
    
    # hyperparameters setting
    neurons = 5
    layers = 2
    act_func = 'tanh'
    numb_of_epochs = 2000
    weight_reg = 0
    
    # store the evaluation results in eval_result
    eval_result = pd.DataFrame(columns = ['opt_function', 'train_acc', 'val_acc'])
         
    
    for opt in optimizers:
        
        opt_function = opt
        
        print('Validating: '+str(opt_function))
        
        # design the model
        model = design_ANN(neurons, layers, act_func, 
                       num_input, num_output, opt, 
                       loss_function, weight_reg)
        
        model.summary()
        
        # train the model
        history = model.fit(X_train, y_train, 
                            batch_size = 32, 
                            epochs = numb_of_epochs, 
                            verbose=1, 
                            validation_data=(X_val, y_val))
        
        # convert the history.history dict to a pandas DataFrame and save it for later use    
        hist_df = pd.DataFrame(history.history)
        
        hist_csv_file = './validation_results/opt/history_'+opt_function+'.csv'
        with open(hist_csv_file, mode='w') as file:
            hist_df.to_csv(file)
        
        # Evaluate the model
        
        train_acc = model.evaluate(X_train, y_train, verbose=0)
        val_acc = model.evaluate(X_val, y_val, verbose=0)    
        print('\ttrain acc: %.3f \tval acc: %.3f' % ( train_acc, val_acc))
        
        eval_result.loc[eval_result.shape[0]] = [opt_function, train_acc, val_acc]
        
    
    # save the result
    with open('./validation_results/opt/result_models.csv', mode='w') as file:
            eval_result.to_csv(file)
    
    # 4. Tune the Layer weight regularizers
    
    # hyperparameters setting
    neurons = 5
    layers = 2
    act_func = 'tanh'
    numb_of_epochs = 1000
    weight_reg = 0
    opt = 'Adamax'
    
    # store the evaluation results in eval_result
    eval_result = pd.DataFrame(columns = ['L2_reg', 'train_acc', 'val_acc'])    
    
    for reg in weight_reg:
       
        print('Validating: '+str(reg))
        
        regularization = reg
        
        # design the model
        model = design_ANN(neurons, layers, act_func, 
                       num_input, num_output, opt, 
                       loss_function, reg)
        
        model.summary()
        
        # train the model
        history = model.fit(X_train, y_train, 
                            batch_size = 32, 
                            epochs = numb_of_epochs, 
                            verbose=1, 
                            validation_data=(X_val, y_val))
        
        # convert the history.history dict to a pandas DataFrame and save it for later use    
        hist_df = pd.DataFrame(history.history)
        
        hist_csv_file = './validation_results/L2_reg/history_'+str(reg)+'.csv'
        with open(hist_csv_file, mode='w') as file:
            hist_df.to_csv(file)
        
        # Evaluate the model
        
        train_acc = model.evaluate(X_train, y_train, verbose=0)
        val_acc = model.evaluate(X_val, y_val, verbose=0)    
        print('\ttrain acc: %.3f \tval acc: %.3f' % ( train_acc, val_acc))
        
        eval_result.loc[eval_result.shape[0]] = [regularization, train_acc, val_acc]
    
    # save the result
    with open('./validation_results/L2_reg/result_models.csv', mode='w') as file:
            eval_result.to_csv(file)
    
    
if __name__ == "__main__":
    main()
       
    