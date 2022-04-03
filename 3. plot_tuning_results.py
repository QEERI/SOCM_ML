# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 11:21:42 2022

@authors: G. Scabbia, A. Abotaleb, A. Sinopoli

        Qatar Environment and Energy Research Institute (QEERI), 
        Hamad Bin Khalifa University (HBKU), Qatar Foundation, 
        P.O. Box 34110, Doha, Qatar
        
        Corresponding authors: asinopoli@hbku.edu.qa

@project-title: Sulphur Oxidative Coupling of Methane process development 
                and its modelling via Machine Learning
    
@file description: script for plotting the result of the hyperparameter tuning process
                   1. Plot the results of the Activation function tuning
                   2. Plot the results of the ANN architecture tuning
                   3. Plot a comparison of the performance of the different
                         architectures and of the best design
                   4. Plot the results of the Optimization function tuning
                   5. Plot the results of the Layer weight regularizers tuning

    Input data is loaded from 
        activation function:    ./validation_results/activation_function/
        architecture:           ./validation_results/architecture/
        optimizaer:             ./validation_results/opt/
        regularization term:    ./validation_results/L2_reg/
    Output figures are saved in ./Figures/


"""

# import libraries
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib.ticker as mticker
#from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def main():
    
    ## Plot activation_functions tuning results
    list_activation_functions = ['sigmoid', 'tanh', 'relu', 'LeakyReLU', 
                                 'PReLU', 'elu']
    
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(8,5))
    
    colors = sns.color_palette("Set2")
    
    c = 0
    
    for function in list_activation_functions:
        
        history = pd.read_csv('./validation_results/activation_function/history_'+str(function)+'.csv')
        
        ax1.plot(np.sqrt(history['val_loss'])*100, label=function, color=colors[c])
        ax1.set_title('Validation losses', fontsize=14)
        ax1.set_ylabel('loss (Total RMSE [%])', fontsize=14)
        ax1.set_xlabel('epoch', fontsize=14)
        ax1.set_xlim([-20, 300])
        ax1.legend(fontsize=14)
        for t in ax1.xaxis.get_major_ticks(): t.label.set_fontsize(13)
        for t in ax1.yaxis.get_major_ticks(): t.label.set_fontsize(13)
        
        if function == 'tanh':
            ax2.plot(np.sqrt(history['loss'])*100, label='training', color='#2f4b7c')
            ax2.plot(np.sqrt(history['val_loss'])*100, label='validation', color='#ffa600')
            ax2.set_title('Activation function: tanh',fontsize=14)
            ax2.set_ylabel('loss (Total RMSE [%])', fontsize=14)
            ax2.set_xlabel('epoch', fontsize=14)
            ax2.set_ylim([2, 10])
            ax2.set_xlim([-10, 300])
            ax2.legend(fontsize=14)
    
            for t in ax2.xaxis.get_major_ticks(): t.label.set_fontsize(13)
            for t in ax2.yaxis.get_major_ticks(): t.label.set_fontsize(13)
            
        c = c + 1
        
    plt.tight_layout()
    plt.savefig('./Figures/loss_activation_functions.svg')
    plt.savefig('./Figures/loss_activation_functions.png')
    
    ## Plot ANN architecture tuning results
    
    num_hidden_layers = [1, 2, 3]
    num_of_neurons = [5, 6, 7]
    
    fig, ax = plt.subplots(1,3, figsize=(13,5))
    
    colors = ["#003f5c", "#bc5090", "#ffa600"]
    
    for layers in num_hidden_layers:
        
        ax1 = ax[layers-1].inset_axes([0.5, 0.5, 0.45, 0.45])
        
        c = 0
        print('\n')
        
        for nodes in num_of_neurons:
            
            function = str(layers)+'_'+str(nodes)
        
            history = pd.read_csv('./validation_results/architecture/history_'+str(function)+'.csv')
            
            mid_value = round(np.mean(np.sqrt(history['val_loss'].iloc[995:1005])*100),2)
            last_value = round(np.mean(np.sqrt(history['val_loss'].iloc[-10:-1])*100),2)
            
            print(function+': 1000 - '+str(mid_value) + '  2000 - '+str(last_value))
            
            ax[layers-1].plot(np.sqrt(history['val_loss'])*100, color=colors[c])
            ax[layers-1].set_title('Validation losses: '+str(layers)+' hidden layer', fontsize=14)
            if layers == 1:
                ax[layers-1].set_ylabel('loss (Total RMSE [%])', fontsize=14)
            ax[layers-1].set_xlabel('epoch', fontsize=14)            
            
            # zoom window
            ax1.plot(np.sqrt(history['val_loss'])*100, color=colors[c])
            
            ax[layers-1].set_ylim([-0.9, 10])
            ax[layers-1].set_xlim([-50, 1500])
            
            if layers == 1:
                ax1.set_xlim([0, 900])
                ax1.set_ylim([0.2, 3.8])
                
            elif layers == 2:
                ax1.set_ylim([0.2, 3.8])
                ax1.set_xlim([0, 400])
                
            elif layers == 3:
                ax1.set_ylim([0.2, 3.8])
                ax1.set_xlim([0, 400])
                
            for t in ax1.xaxis.get_major_ticks(): t.label.set_fontsize(11)
            for t in ax1.yaxis.get_major_ticks(): t.label.set_fontsize(11)
        
            ax[layers-1].indicate_inset_zoom(ax1, edgecolor="black", alpha=0.2)
            
            c = c + 1
            
        ax[layers-1].legend(['5 nodes', '6 nodes', '7 nodes'], ncol=3, loc=8, columnspacing=0.6, fontsize=11)
        
        for t in ax[layers-1].xaxis.get_major_ticks(): t.label.set_fontsize(12)
        for t in ax[layers-1].yaxis.get_major_ticks(): t.label.set_fontsize(12)
        
    plt.tight_layout()
    plt.savefig('./Figures/loss_architecture.svg')
    plt.savefig('./Figures/loss_architecture.png')
    
    ## Plot architecture comparison and best design performance
    
    Ni = 3
    No = 8
    numb_parameters = []
    
    for L in [1,2,3]:
        for n in [5, 6, 7]:
            n_weights = Ni*n + sum([n*n for l in range(1,L)]) + n*No
            n_bias = n*L + No
            tot = n_weights + n_bias
            numb_parameters.append(tot)
    
    label = ['1_5', '1_6', '1_7', '2_5', '2_6', '2_7', '3_5', '3_6', '3_7']
    
    eval_result = pd.read_csv('./validation_results/architecture/result_models.csv')
    
    eval_result.val_acc = np.sqrt(eval_result.val_acc)*100
    
    fig2, ax2 = plt.subplots(1,2, figsize=(9,5))
    
    z = np.polyfit(numb_parameters, eval_result.val_acc.dropna().values, 3)
    p = np.poly1d(z)
    x = range(numb_parameters[0], numb_parameters[-1]+1)
    ax2[0].plot(x, p(x),"r--", alpha=0.3)
    
    ax2[0].scatter(numb_parameters, eval_result.val_acc.dropna().values)
    i = 0
    for x,y in zip(numb_parameters, eval_result.val_acc.dropna().values):
        ax2[0].annotate(label[i], # this is the text
             (x,y), # this is the point to label
             textcoords="offset points", # how to position the text
             xytext=(15,3), # distance from text to points (x,y)
             ha='center', # horizontal alignment can be left, right or center
             fontsize=11)          
        i = i + 1
    
    ax2[0].set_title('Comparison: \nmodel complexity vs validation MSE', fontsize=14)
    ax2[0].set_ylabel('loss (Average rRMSE [%])', fontsize=15)
    ax2[0].set_xlabel('number of model \ntrainable parameters', fontsize=14)
    ax2[0].set_xlim([50, 225])
    for t in ax2[0].xaxis.get_major_ticks(): t.label.set_fontsize(13)
    for t in ax2[0].yaxis.get_major_ticks(): t.label.set_fontsize(13)
    #ax2[0].set_ylim([0, 1e-3])
    
    layers = 2
    nodes = 5
    
    function = str(layers)+'_'+str(nodes)
        
    history = pd.read_csv('./validation_results/architecture/history_'+str(function)+'.csv')
        
    ax2[1].plot(np.sqrt(history['val_loss'])*100, label='validation', color='#ffa600')
    ax2[1].plot(np.sqrt(history['loss'])*100, label='training', color='#2f4b7c')
    ax2[1].set_title('ANN: \n2 hidden layers and 5 neurons', fontsize=14)
    ax2[1].set_ylabel('loss (Average rRMSE [%])', fontsize=15)
    ax2[1].set_xlabel('epoch', fontsize=14)
    #ax2[1].set_yscale("log")
    ax2[1].set_xlim([-10, 400])
    ax2[1].set_ylim([0, 5])
    ax2[1].legend( fontsize=14)
    for t in ax2[1].xaxis.get_major_ticks(): t.label.set_fontsize(13)
    for t in ax2[1].yaxis.get_major_ticks(): t.label.set_fontsize(13)
    
    fig2.tight_layout()
    fig2.savefig('./Figures/loss_best_architecture.svg')
    fig2.savefig('./Figures/loss_best_architecture.png')
    
    
    ## Plot Optimization function tuning results
    
    colors = sns.color_palette("Set2")
    
    optimizer = ['SGD', 'RMSprop', 'Adam', 'Adamax', 'Nadam']    
    
    fig, ax = plt.subplots(3,2, figsize=(10,14))
    
    i = 0
    j = 0
    
    for function in optimizer:
        
       
        history = pd.read_csv('./validation_results/opt/history_'+str(function)+'.csv')
        
        ax[i, j].plot(np.sqrt(history['val_loss'])*100, label='validation', color='#ffa600')
        ax[i, j].plot(np.sqrt(history['loss'])*100, label='training', color='#2f4b7c')
        ax[i, j].set_title(function, fontsize=15)
        ax[i, j].set_ylabel('loss (rRMSE [%])', fontsize=14)
        ax[i, j].set_xlabel('epoch', fontsize=14)
        ax[i, j].set_ylim([0, 5])
        ax[i, j].legend(loc='best', fontsize=14)
        
        mid_value = np.mean((np.sqrt(history['val_loss'])*100).iloc[994:1004])
        
        ax[i, j].annotate(round(mid_value,2), #'{:.1e}'.format(mid_value), # this is the text
             (1000,mid_value), # this is the point to label
             textcoords="offset points", # how to position the text
             xytext=(-10,-20), # distance from text to points (x,y)
             ha='center',
             size=12, va="center", color='blue', # horizontal alignment can be left, right or center
             fontsize=13)
        ax[i, j].plot(1000, mid_value,'o', ms=6, color='blue')
        
        last_value = np.mean((np.sqrt(history['val_loss'])*100).iloc[-5:-1])    
        
        ax[i, j].annotate(round(last_value,2), #'{:.1e}'.format(last_value), # this is the text
             (2000,last_value), # this is the point to label
             textcoords="offset points", # how to position the text
             xytext=(-10,-25), # distance from text to points (x,y)
             ha='center',
             size=12, va="center", color='red', # horizontal alignment can be left, right or center
             fontsize=13)
        ax[i, j].plot(2000, last_value,'o', ms=6, markevery=[-1], color='red')
        
        for t in ax[i, j].xaxis.get_major_ticks(): t.label.set_fontsize(11.5)
        for t in ax[i, j].yaxis.get_major_ticks(): t.label.set_fontsize(13)
           
        j = j + 1
        
        if j == 2:
            i = i + 1
            j = 0
            
    n = 0
    for function in optimizer:
        
        history = pd.read_csv('./validation_results/opt/history_'+str(function)+'.csv')
        
        ax[2, 1].plot(np.sqrt(history['val_loss'])*100, label=function, color=colors[n])
        
        n = n + 1
        
    ax[2, 1].set_ylabel('loss (rRMSE [%])', fontsize=14)
    ax[2, 1].set_xlabel('epoch', fontsize=14)
    ax[2, 1].set_ylim([0.5,3])
    ax[2, 1].legend(loc='best', ncol=3, fontsize=11)
    ax[2, 1].set_title('Comparison - validation loss', fontsize=15)
    
    for t in ax[2, 1].xaxis.get_major_ticks(): t.label.set_fontsize(11.5)
    for t in ax[2, 1].yaxis.get_major_ticks(): t.label.set_fontsize(13)
    
    fig.tight_layout()
    fig.savefig('./Figures/loss_opt.svg')
    fig.savefig('./Figures/loss_opt.png')
    
    ## Plot regularization tuning results

    f = mticker.ScalarFormatter(useMathText=True)
    f.set_powerlimits((-3,3))
    
    # regularization
    weight_reg = [0, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(8,5))
    
    import seaborn as sns
    colors = sns.color_palette("Set2")
    
    c = 0
    
    for reg in weight_reg:
        
        history = pd.read_csv('./validation_results/L2_reg/history_'+str(reg)+'.csv')
        
        if reg > 0:
            ax1.plot(np.sqrt(history['loss'])*100, label="${}$".format(f.format_data(reg)) , color=colors[c])
        else:
            ax1.plot(np.sqrt(history['loss'])*100, label="0" , color=colors[c])
        ax1.set_title('Training losses', fontsize=15)
        ax1.set_ylabel('loss (Total RMSE [%])', fontsize=14)
        ax1.set_xlabel('epoch', fontsize=14)
        ax1.set_ylim([0.5, 10])
        ax1.legend(title='L2 penalty', title_fontsize=12.5, fontsize=12.5)
        
        for t in ax1.xaxis.get_major_ticks(): t.label.set_fontsize(13)
        for t in ax1.yaxis.get_major_ticks(): t.label.set_fontsize(13)
        
        if reg == 0:
            ax2.plot(np.sqrt(history['val_loss'])*100, label='validation', color='#ffa600')
            ax2.plot(np.sqrt(history['loss'])*100, label='training', color='#2f4b7c')
            ax2.set_title('No L2 weight regularization', fontsize=15)
            ax2.set_ylabel('loss (Total RMSE [%])', fontsize=14)
            ax2.set_xlabel('epoch', fontsize=14)

            ax2.set_ylim([0.5, 4])
            ax2.legend(fontsize=14)
                
            mid_value = np.mean(np.sqrt(history['val_loss'].iloc[500-6:500+4])*100)
            
            ax2.annotate(round(mid_value,2), #'{:.1e}'.format(mid_value), # this is the text
                 (500,mid_value), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(-10,-15), # distance from text to points (x,y)
                 ha='center',
                 size=12, va="center", color='blue') # horizontal alignment can be left, right or center
    
            ax2.plot(500, mid_value,'o', ms=6, color='blue')
            
            last_value = np.mean(np.sqrt(history['val_loss'].iloc[-5:-1])*100)
            
            ax2.annotate(round(last_value,2), #'{:.1e}'.format(last_value), # this is the text
                 (1000,last_value), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(-10,-15), # distance from text to points (x,y)
                 ha='center',
                 size=12, va="center", color='red', # horizontal alignment can be left, right or center
                 fontsize=13)
            ax2.plot(np.sqrt(history['loss'])*100,'o', ms=6, markevery=[-1], color='red')
            
            for t in ax2.xaxis.get_major_ticks(): t.label.set_fontsize(13)
            for t in ax2.yaxis.get_major_ticks(): t.label.set_fontsize(13)
            
        c = c + 1
        
    plt.tight_layout()
    plt.savefig('./Figures/loss_reg.svg')
    plt.savefig('./Figures/loss_reg.png')
    
    
if __name__ == "__main__":
    main()