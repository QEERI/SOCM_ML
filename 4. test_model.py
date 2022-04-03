# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 11:36:39 2022

@authors: G. Scabbia, A. Abotaleb, A. Sinopoli

        Qatar Environment and Energy Research Institute (QEERI), 
        Hamad Bin Khalifa University (HBKU), Qatar Foundation, 
        P.O. Box 34110, Doha, Qatar
        
        Corresponding authors: asinopoli@hbku.edu.qa

@project-title: Sulphur Oxidative Coupling of Methane process development 
                and its modelling via Machine Learning
    
@file description: script for testing the model performance
                   1. Load the data
                   2. Set the model and design the ANN
                   3. Train the model
                   4. Evaluate the model and plot the results

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
from sklearn.metrics import r2_score

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
        
    ## Model settings and desing
    
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
    
    model = design_ANN(num_of_neurons, 
                       num_hidden_layers, 
                       activation_func, 
                       num_input, 
                       num_output, 
                       opt, 
                       loss_function, 
                       weight_reg)
        
    ## print Model summary
    print(model.summary())
    
    ## Model training

    history = model.fit(X_train, y_train, 
                        batch_size = 32, 
                        epochs = numb_of_epochs, 
                        verbose=1, 
                        validation_data=(X_val, y_val))
    
    ## evaluate the model and plot the results
    
    train_acc = model.evaluate(X_train, y_train, verbose=0)
    val_acc = model.evaluate(X_val, y_val, verbose=0)
    test_acc = model.evaluate(X_test, y_test, verbose=0)
    
    print('Train: %.5f, Validation: %.5f, Test: %.5f' % (train_acc, val_acc, test_acc))
    
    print('RMSE Train: %.5f, Validation: %.5f, Test: %.5f' % (np.sqrt(train_acc)*100, 
                                                              np.sqrt(val_acc)*100, 
                                                              np.sqrt(test_acc)*100))
    
    # store the predicted values
    pred = pd.DataFrame(model.predict(X_test), columns=y_test.columns, index=y_test.index)
    
    # print the results
    out_losses = []
    for col in y_test.columns:
        loss = np.sqrt(mean_squared_error(y_test[col].values, pred[col].values))*100
        out_losses.append(loss)
        print(col+': '+str(round(loss,2)))
        
    out_losses = np.array(out_losses)
    
    # plot the results
    fig, ax = plt.subplots(1,2, figsize=(8,5))
    
    # loss function
    ax[0].plot(np.sqrt(history.history['val_loss'])*100, label='validation', color='#ffa600')
    ax[0].plot(np.sqrt(history.history['loss'])*100, label='train', color='#2f4b7c')
    ax[0].legend(fontsize=14)
    ax[0].set_ylim([0, 5])
    ax[0].set_xlabel('epochs', fontsize=14)
    ax[0].set_ylabel('average rRMSE loss [%]', fontsize=14)
    
    for t in ax[0].xaxis.get_major_ticks(): t.label.set_fontsize(13)
    for t in ax[0].yaxis.get_major_ticks(): t.label.set_fontsize(13)
    
    ax[1].barh(y_test.columns, out_losses, color=sns.color_palette('Set2'))
    ax[1].invert_yaxis()
    ax[1].set_xlabel('rRMSE loss [%]', fontsize=14)
    
    for i, v in enumerate(out_losses):
        ax[1].text(v + 0.1, i+0.1, str(round(v,2)), color='black', fontsize=13)
        
    ax[1].set_xlim([0, 3.5])
    
    for t in ax[1].xaxis.get_major_ticks(): t.label.set_fontsize(13)
    for t in ax[1].yaxis.get_major_ticks(): t.label.set_fontsize(13)
                
    plt.tight_layout()
    plt.savefig('./Figures/result_model.svg')
    plt.savefig('./Figures/result_model.png')
    
    # plot a scatter comparison of predicted vs actual for each output variable
    
    colors = sns.color_palette("colorblind", 
                n_colors=len(X_test['mass_ratio'].unique())) 
    
    fig, ax = plt.subplots(4,2, figsize=(8,15))
    
    i, j = 0, 0
    n= 0
    
    tot_rmse = 0
    
    test = pd.DataFrame(sc_output.inverse_transform(y_test), columns= y_test.columns)
    test_x = pd.DataFrame(sc_input.inverse_transform(X_test), columns= X_test.columns)
    
    prediction = pd.DataFrame(sc_output.inverse_transform(pred), columns= pred.columns)
    
    for variable in input_var:
        
        test[variable] = test_x[variable]
        prediction[variable] = test_x[variable]
    
    avg_R_square = []
    for col in pred.columns:
        
        min_value = np.amin([prediction[col], test[col]])
        max_value = np.amax([prediction[col], test[col]])
        
        mse = mean_squared_error(y_test[col], pred[col])
                
        tot_rmse = tot_rmse + np.sqrt(mse)*100
        
        print(col+' MSE: '+str(round(mse, 5)))
        
        n = 0
        
        R_square = r2_score(test[col], prediction[col])
        avg_R_square.append(R_square)
        
        for MR in np.sort(test['mass_ratio'].unique()): 
            
            ax[i,j].scatter(prediction.loc[prediction['mass_ratio'] == MR, col], 
                                test.loc[test['mass_ratio'] == MR, col], 
                                alpha=0.3, color=colors[n])
            
            n = n + 1
            
        ax[i,j].plot([min_value,max_value], [min_value,max_value], 
                     linestyle='--', color='black')
        #ax[i,j].set_title(col)
        ax[i,j].set_ylabel('actual output')
        ax[i,j].set_xlabel('predicted output')
        
        size = max_value - min_value
        
        ax[i,j].annotate('R$^2$: ' +str(round(R_square*100,2)),
                 (min_value+0.01*size,min_value+0.85*size), 
                 textcoords="offset points",
                 xytext=(0,0),
                 ha='left',
                 size=12, va="center", color='black')
        
        ax[i,j].annotate(col, 
                 (min_value+0.01*size,min_value+0.95*size), 
                 textcoords="offset points", 
                 xytext=(0,0), 
                 ha='left',
                 size=12, va="center", color='black') 
        
        ax[i,j].annotate('rRMSE: ' +str(round(np.sqrt(mse)*100,2))+'%', 
                 (min_value+0.01*size,min_value+0.75*size), 
                 textcoords="offset points", 
                 xytext=(0,0), 
                 ha='left',
                 size=12, va="center", color='black') 
        
        j = j + 1
        if j == 2:
            j = 0
            i = i + 1
            
    
    ax_leg = fig.add_subplot(111)
    ax_leg.axis('off')
    n=0
    for MR in np.sort(test['mass_ratio'].unique()): 
        ax_leg.scatter(np.NaN, np.NaN, s=50, color=colors[n], 
                       label=str(round(MR,2)))
        n=n+1
    ax_leg.legend(title = 'Mass ratio', ncol=5, bbox_to_anchor=(0.5, -0.105),
                  loc='lower center')
    fig.tight_layout()
    
    #print('\nFinal error: '+str(round(tot_mse, 5)))
    print('rRMSE: '+str(round(tot_rmse/num_output, 2)))
    print('avg R_square: '+str(round(np.mean(np.array(avg_R_square)*100), 2)))
    
    
    plt.savefig('./Figures/scatter_comparison.svg')
    plt.savefig('./Figures/scatter_comparison.png')
    
    
    ## plot a set of images to check the effect of a single input variable
    
    var = 'temperature' ## SET THE INPUT VARIABLE TO BE TESTED HERE
    # sect between temperature and pressure
        
    if var == 'temperature':
        x_label = 'pressure'
        x_label_str = 'pressure (bar)'
        sel_value = 800.0
    else :
        x_label = 'temperature'
        x_label_str = 'temperature (°C)'
        sel_value = 1.0
            
    # select data for plot
    
    X_train_base = pd.DataFrame(sc_input.inverse_transform(X_train), 
                                columns= X_train.columns, index=X_train.index)
    y_train_base = pd.DataFrame(sc_output.inverse_transform(y_train), 
                                columns= y_train.columns, index=y_train.index)
            
    X_test_base = pd.DataFrame(sc_input.inverse_transform(X_test), 
                               columns= X_test.columns, index=X_test.index)
    y_test_base = pd.DataFrame(sc_output.inverse_transform(y_test), 
                               columns= y_test.columns, index=y_test.index)
    
    # prediction on full dataset
    
    X_data = X_train.append(X_test, ignore_index=False)
    
    X_data_base = X_train_base.append(X_test_base, ignore_index=False)
    
    pred = pd.DataFrame(sc_output.inverse_transform(model.predict(X_data)), 
                        columns= y_test.columns, index=X_data.index)
    
    train_data = y_train_base.copy(deep=True)
    test_data = y_test_base.copy(deep=True)
    
    colors = sns.color_palette("colorblind", 
                n_colors=len(X_data_base.mass_ratio.unique())) 
    
    
    for variable in input_var:
        
        train_data[variable] = X_train_base[variable]
        test_data[variable] = X_test_base[variable]
        pred[variable] = X_data_base[variable]
    
    i = 0
    j = 0
    
    fig, ax = plt.subplots(4,2, figsize=(12,15))
    
    for col in y_test.columns:
        
        n = 0
        
        for MR in np.sort(train_data['mass_ratio'].unique()):
    
            #train = train_data.loc[(train_data['mass_ratio'] == MR)&(train_data[var] == sel_value)]
            test = test_data.loc[(test_data['mass_ratio'] == MR)&(test_data[var] == sel_value)]
            prediction = pred.loc[(pred['mass_ratio'] == MR)&(pred[var] == sel_value)]
                
            ax[i,j].scatter(test[x_label], 
                             test[col], color=colors[n], marker='x')
    
            ax[i,j].plot(prediction.sort_values(by=x_label)[x_label], 
                         prediction.sort_values(by=x_label)[col], color=colors[n])
        
            n = n + 1
        
        ax[i,j].set_xlabel(x_label_str)
        ax[i,j].set_ylabel('Comparison')
        
        ax[i,j].set_title(col)
        
        j = j + 1
        if j == 2:
            j = 0
            i = i + 1
            
        
    ax_leg = fig.add_subplot(111)
    ax_leg.axis('off')
    ax_leg.plot(np.NaN, np.NaN, 'x', color='k', label='test obs.')
    ax_leg.plot(np.NaN, np.NaN, color='black', label='predicted')
    n=0
    for MR in np.sort(train_data['mass_ratio'].unique()): 
        ax_leg.plot(np.NaN, np.NaN, 's', color=colors[n], label=str(round(MR,2)))
        n=n+1
    ax_leg.legend(title = 'Mass ratio', ncol=6, 
                  bbox_to_anchor=(0.5, -0.105), loc='lower center')
    fig.tight_layout()
    
    plt.savefig('./Figures/result_'+x_label+'.svg')
    plt.savefig('./Figures/result_'+x_label+'.png')


if __name__ == "__main__":
    main()
    
    
  