# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 13:01:23 2022

@authors: G. Scabbia, A. Abotaleb, A. Sinopoli

        Qatar Environment and Energy Research Institute (QEERI), 
        Hamad Bin Khalifa University (HBKU), Qatar Foundation, 
        P.O. Box 34110, Doha, Qatar
        
        Corresponding authors: asinopoli@hbku.edu.qa

@project-title: Sulphur Oxidative Coupling of Methane process development 
                and its modelling via Machine Learning
    
@file description: script for testing the ANN as interpolation tool to 
                   find optimal process conditions
                   1. Load the data
                   2. Filter the data within the real operating conditions
                   3. Design the model
                   4. Train the model
                   5. Test ANN as interpolation tool with 100 points
                   6. check the effect of the number of points in the interpolation accuracy 

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

from sklearn.model_selection import train_test_split

from scipy.stats import spearmanr

from pickle import load


# global vars
RAND_STATE = 1           # Controls the shuffling applied to the data before 
                         #    applying the split. Pass an int for reproducible 
                         #    output across multiple function calls.

def calc_score(df):
    
    return (df['H2S_conv']*0.3 + 
              df['CS2_conv']*0.1 +
              df['naphthalene']*0.15 +
              df['p_Xylene']*0.15 +
              df['toluene']*0.15 +
              df['benzene']*0.15) / (df['H2S'] + df['methane'])


def filter_opt_cond(data_array, sc_input):
    
    # [mins], [maxs]
    #[temperature, pressure, mass_ratio]
    min_values, max_values = sc_input.transform([[500, 1, 1], [1000, 5, 10]])
    
    result = []
    
    for data in data_array:
        ## select operational conditions
        
        data = data.loc[(data['temperature'] >= min_values[0])&(data['temperature'] <= max_values[0])]

        data = data.loc[(data['pressure'] >= min_values[1])&(data['pressure'] <= max_values[1])]

        data = data.loc[(data['mass_ratio'] >= min_values[2])&(data['mass_ratio'] <= max_values[2])]
        
        result.append(data)

    return result



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



def plot_3Ddata(df, ax, title):
    
    ### subplot 1: 3D scatterplot of the dataset
    mapp = ax.scatter(df['temperature'], df['mass_ratio'], df['pressure'], 
                c=df['score'], alpha=0.5)

    # change the fontsize of the ticks
    for t in ax.xaxis.get_major_ticks(): t.label.set_fontsize(11)
    for t in ax.yaxis.get_major_ticks(): t.label.set_fontsize(11)
    for t in ax.zaxis.get_major_ticks(): t.label.set_fontsize(11)

    # add axis labels and change fontsize
    ax.set_xlabel('Temperature', fontsize=12)
    ax.set_ylabel('Mass_ratio', fontsize=12)
    ax.set_zlabel('Pressure', fontsize=12)
    
    ax.set_title(title)
    
    return mapp



def split_data(X_data, y_data, rand_state, DIM):
    
    #train - test split
    test_val_size = int(X_data.shape[0]*0.2)

    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, 
               test_size=test_val_size, random_state=rand_state)

    # take a portion of the data (n=DIM)
    X_train = X_train.sample(n=DIM, random_state=rand_state)
    y_train = y_train.loc[X_train.index]

    #print('Train Subset size: ' +str(X_train.shape[0]))
    #print('Test_set size: ' +str(X_test.shape[0]))
    
    return X_train, X_test, y_train, y_test



def test_classification(r_res, best_points, data_threshold, ML_threshold):
    
    tot_samples = r_res['sim_score'].shape[0]

    false_pos = best_points.loc[(best_points['sim_score']<data_threshold)&
                           (best_points['ML_score']>=ML_threshold)].shape[0]

    false_neg = best_points.loc[(best_points['sim_score']>=data_threshold)&
                           (best_points['ML_score']<ML_threshold)].shape[0]

    true_pos = best_points.loc[(best_points['sim_score'] >= data_threshold)&
                           (best_points['ML_score']>=ML_threshold)].shape[0]
    
    pos = r_res.loc[r_res['sim_score']>=data_threshold].shape[0]
    
    neg = tot_samples - pos
    
    pred_pos = r_res.loc[r_res['ML_score']>=ML_threshold].shape[0]
    
    pred_neg = tot_samples - pred_pos

    true_neg = tot_samples - false_pos - false_neg - true_pos
    
    return pos, neg, pred_pos, pred_neg, false_pos, false_neg, true_pos, true_neg



def main():
    
    
    input_var = ['temperature', 'pressure', 'mass_ratio']
    
    output_var = ['H2S_conv', 'CS2_conv', 'naphthalene', 'p_Xylene', 
                  'toluene', 'benzene', 'H2S', 'methane']
    
    ## Load the data
       
    original_train_x = pd.read_csv('./input data/X_train.csv') #.sample(n=100, random_state=1)
    original_train_y = pd.read_csv('./input data/y_train.csv') #.loc[X_train.index]
    
    original_val_x = pd.read_csv('./input data/X_val.csv')
    original_val_y = pd.read_csv('./input data/y_val.csv')
    
    original_test_x = pd.read_csv('./input data/X_test.csv')
    original_test_y = pd.read_csv('./input data/y_test.csv')
    
    # recombine the data in one dataframe (X and y)
    X_data = original_train_x.append(original_val_x, ignore_index=False, verify_integrity=True)
    X_data = X_data.append(original_test_x, ignore_index=False, verify_integrity=True)
    
    y_data = original_train_y.append(original_val_y, ignore_index=False, verify_integrity=True)
    y_data = y_data.append(original_test_y, ignore_index=False, verify_integrity=True)
    
    # load scaling functions
    sc_input = load(open('./input data/sc_input.pkl', 'rb'))
    sc_output = load(open('./input data/sc_output.pkl', 'rb'))
    
    print('Dataset size: ' +str(X_data.shape[0]))
    
    
    ## Filter the data within the real operating conditions
    
    # data boundaries
    min_values, max_values = [[500, 1, 1], [1000, 5, 10]]
    min_values_tx, max_values_tx = sc_input.transform([[500, 1, 1], [1000, 5, 10]])
    
    i = 0
    for var in input_var:
        X_data = X_data.loc[(X_data[var] >= min_values_tx[i])&
                                (X_data[var] <= max_values_tx[i]) ]    
        i = i + 1
        
    y_data = y_data.loc[X_data.index]
    
    print('Filtered dataset size: ' +str(X_data.shape[0]))
    
    ## split the data (DIM is the subset dimension for the training set. 
    #      Test set size = 20% of the data)
    X_train, X_test, y_train, y_test = split_data(X_data, y_data, 
                                                  rand_state=RAND_STATE, 
                                                  DIM=100)
    
    ## Design the model

    nodes = 5
    layers = 2
    act_func = 'tanh'
    num_input = X_train.shape[1]
    num_output = y_train.shape[1]
    opt = 'Adamax'
    loss_function = 'mean_squared_error'
    weight_reg = 0
    numb_of_epochs = 1000
    
    model = design_ANN(nodes, layers, act_func, 
               num_input, num_output, opt, loss_function, weight_reg)
    
    model.summary()
    
    ## Train the model
    history = model.fit(X_train, y_train, 
                        batch_size = 32, 
                        epochs = numb_of_epochs, 
                        verbose=1)
    
    # create a search space between the limits and test the ANN on this set
    step = 10 # resolution for the interpolated points
    
    search_space = pd.DataFrame(np.array(np.meshgrid(
        np.linspace(min_values_tx[0], max_values_tx[0], step),
        np.linspace(min_values_tx[1], max_values_tx[1], step),
        np.linspace(min_values_tx[2], max_values_tx[2], step)
        )).T.reshape(-1,3), columns=input_var)
    
    # use the model to predict the output for each point in the search_space
    y_pred = pd.DataFrame(model.predict(search_space), columns=y_test.columns)
    
    # create two datasets as proxy copies of the simulated data used for plotting
    sim_all = y_data.copy(deep=True)
    sim_sub = y_train.copy(deep=True)
    
    # calculate score and add it as column in the dataframe
    sim_all['score'] = calc_score(sim_all)
    sim_sub['score'] = calc_score(sim_sub)
    y_pred['score'] = calc_score(y_pred)
    
    # add input var information
    for var in input_var:
        sim_all[var] = X_data[var]
        sim_sub[var] = X_train[var]
        y_pred[var] = search_space[var]
    
    # transform the input variables back to their original levels
    sim_all[input_var] = sc_input.inverse_transform(sim_all[input_var])
    sim_sub[input_var] = sc_input.inverse_transform(sim_sub[input_var])
    y_pred[input_var] = sc_input.inverse_transform(y_pred[input_var])
    
    fig = plt.figure(figsize=(15,4))
    ax1 = fig.add_subplot(1, 4, 1,projection="3d")
    mapp = plot_3Ddata(sim_all, ax1, 'a) original simulated data')
    ax2 = fig.add_subplot(1, 4, 2,projection="3d")
    plot_3Ddata(sim_sub, ax2, 'b) random sub-set n=100')
    ax3 = fig.add_subplot(1, 4, 3,projection="3d")
    plot_3Ddata(y_pred, ax3, 'c) ML interpolated data')
    
    #ax4 = fig.add_subplot(1, 4, 4)
    cbar_ax = fig.add_axes([0.785, 0.15, 0.015, 0.7])
    plt.colorbar(mapp, cax=cbar_ax)
    
    fig.tight_layout(w_pad=4)
    
    plt.savefig('./Figures/sim_ML_opt.svg')
    plt.savefig('./Figures/sim_ML_opt.tiff')
    
    # test the effect of different dimensions DIM
    
    result = pd.DataFrame(columns=[
        'round', 'DIM', 'len_opt', 'mean', 'median', 'Q1', 'Q3',  
        'false_neg','FNR', 'false_pos', 'FDR', 'precision', 'recall',
        's_coeff', 'p_value', 'accuracy', 'F_score'])
    
    for DIM in [100, 250, 500, 1000]:
        
        print('Dim: '+str(DIM))
    
        for r in range(100):
    
            print('  Round: '+str(r+1))
    
            r_res = pd.DataFrame() # where to save the result comparison
    
            # split the data
            X_train, X_test, y_train, y_test = split_data(X_data, y_data, 
                                        rand_state=r, DIM=DIM)
    
            # reset the model
            model = design_ANN(nodes, layers, act_func, 
                   num_input, num_output, opt, loss_function, weight_reg)
    
            #model.summary()
    
            # train the model
            history = model.fit(X_train, y_train, 
                                batch_size = 32, 
                                epochs = numb_of_epochs, 
                                verbose=0)
    
            # calculate score for the simulated subset
            r_res['sim_score'] = calc_score(y_test)
    
            # ML prediction
            y_pred = pd.DataFrame(model.predict(X_test), columns=y_test.columns)
    
            # calculate score for the ML prediction
            r_res['ML_score'] = calc_score(y_pred).values
    
            # filter out the points with the best score (> 95 percentile)
            best_points = r_res.loc[r_res['sim_score'] >= np.percentile(calc_score(y_data), 95)]
    
            # calculate absolute percentage error
            best_points_APE = (np.abs(best_points['sim_score']-
                                      best_points['ML_score'])/best_points['sim_score']).values*100
    
            # calculate spearman coeff
            coeff, p_value = spearmanr(best_points['sim_score'], best_points['ML_score'])
            
            # assess model accuracy in finding optimal conditions        
            pos, neg, pred_pos, pred_neg, false_pos, false_neg, true_pos, true_neg = test_classification(
                        r_res, 
                        best_points,
                        data_threshold=np.percentile(calc_score(y_data), 95), 
                        ML_threshold=np.percentile(calc_score(y_pred), 95))
    
            FNR = false_neg/pos       # False negative rate
            recall = 1 - FNR
            FDR = false_pos/pred_pos  # False discovery rate
            precision = 1 - FDR
            F_score = 2*precision*recall/(precision+recall)
    
            accuracy = (true_pos + true_neg)/(pos + neg)
            
            # store the results
            result.loc[result.shape[0]] = [r, DIM, best_points.shape[0], 
                                            np.mean(best_points_APE), 
                                            np.median(best_points_APE),
                                            np.percentile(best_points_APE,25), 
                                            np.percentile(best_points_APE,75),
                                            false_neg, FNR, false_pos, 
                                            FDR, precision, recall,
                                            coeff, p_value, accuracy, F_score]
            
    ## plot and save the results
    
    # group the results
    result_group = result.groupby(by='DIM', as_index=False).median()
    result_group['DIM'] = result_group['DIM'].astype('int')
    result['DIM'] = result['DIM'].astype('int')
    
    fig, ax = plt.subplots(1,2, figsize=(10,5))
    ax[0].errorbar(result_group.DIM, result_group['median'], yerr = [result_group['median']-result_group['Q1'],
                                                                  result_group['Q3']-result_group['median']], 
                 fmt='.',ecolor = 'black',color='grey', markersize=15, capsize=3, capthick=1, elinewidth=1)
    ax[0].plot(result_group.DIM, result_group['median'], alpha=0.6, c='orange', linewidth=1)
    #ax.set_xscale('log')
    for t in ax[0].xaxis.get_major_ticks(): t.label.set_fontsize(13)
    for t in ax[0].yaxis.get_major_ticks(): t.label.set_fontsize(13)
    ax[0].set_xlabel('Training set size (subset size)', fontsize=14)
    ax[0].set_ylabel('Median error on score prediction [%]', fontsize=14)
    
    flierprops=dict(marker='.', markersize=6)
    meanpointprops = dict(marker='D', markeredgecolor='None',
                          markerfacecolor='navy', alpha=0.5)
    
    sns.boxplot(data=result, x='DIM', y='F_score', ax=ax[1], palette='Set2', showmeans=True, 
                flierprops=flierprops, meanprops=meanpointprops)
    for t in ax[1].xaxis.get_major_ticks(): t.label.set_fontsize(13)
    for t in ax[1].yaxis.get_major_ticks(): t.label.set_fontsize(13)
    ax[1].set_xlabel('Training set size (sub-set size)', fontsize=14)
    ax[1].set_ylabel('F-score', fontsize=14)
    
    fig.tight_layout()
    
    plt.savefig('./Figures/opt_group.svg')
    plt.savefig('./Figures/opt_group.png')
    
    result_group.to_csv('result_group.csv')
    result.to_csv('result.csv')
    
    
    
    
if __name__ == "__main__":
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    