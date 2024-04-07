#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 03:02:30 2024

@author: magdalenelim
"""


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np 
import random 
from itertools import combinations
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import log_loss, accuracy_score,  mean_squared_error,confusion_matrix
from sklearn.decomposition import PCA




np.random.seed(8)


# Read the data file into a DataFrame
df = pd.read_csv('/Users/magdalenelim/Desktop/parkinsons/parkinsons.data') 

plt.clf()
plt.cla() 

dir_ = '/Users/magdalenelim/Desktop/he4903proj/'

df['name'] = df['name'].str[10:12]
df.name = df.name.astype(int) 


df = df.drop([ 'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'spread1', 'spread2'], axis=1) 
df['status'] = pd.Categorical(df['status'])


# ===========================Features Scale===========================#

def plot_feature_variation_boxplot(): 
    df_ = df.drop([ 'name'], axis=1) 

    sns.set(style="whitegrid", font_scale=1.2)
    
    df_=df.copy() 
    df_ = df_.drop(['name', 'status', 'spread1', 'spread2'], axis=1) 
    plt.figure(figsize=(15, 10))  # Adjust the figure size as needed
    for i, column in enumerate(df_.columns):
        plt.subplot(4, 5, i+1)  # Adjust subplot layout if necessary
        sns.boxplot(y=df[column])
        plt.ylabel('')
        plt.title(column)
        plt.tight_layout()
    plt.rcParams.update({'axes.titlesize': 'medium'})
    plt.savefig(dir_ + 'featuresscaleboxplot.pdf', bbox_inches='tight')
    
def plot_feature_variation_gaussianplot(): 
    sns.set(style="whitegrid", font_scale=1.2)
    
    plt.figure(figsize=(15, 10))  # Adjust the figure size as needed
    for i, column in enumerate(df.columns):
        plt.subplot(4, 5, i+1)  # Adjust subplot layout if necessary
        sns.kdeplot(data=df.loc[df['status'] == 1, column] , shade=False, color="red")
        sns.kdeplot(data=df.loc[df['status'] == 0, column] , shade=False, color="green")
        plt.ylabel('')
        plt.title(column)
        plt.tight_layout()
    plt.rcParams.update({'axes.titlesize': 'medium'})
    
    
    plt.savefig(dir_ + 'featuresscaleplot2.pdf', bbox_inches='tight')
    
def preproc(df):    
    for col in df.columns: 
        if col != 'name': 
            df[col] = (df[col] - df.col.mean()) / df.col.std() 
    return df 
    

def split(df, test_index ): 
    test_df = df[df['name'].isin(test_index)]
    train_df = df[~df['name'].isin(test_index)]
    return test_df.drop('status', axis=1), test_df['status'], train_df.drop('status',axis=1), train_df['status'] 
    

def split_loso(df, test_index ): 
    test_df = df[df['name'] == test_index ] 
    train_df = df[df['name'] != test_index]
    return test_df.drop(['status','name'], axis=1), test_df['status'], train_df.drop(['status','name'],axis=1), train_df['status'] 

    
# ===========================Correlation ==========================#

def stats_from_model(model, x_test, y_test ): 
    y_pred = model.predict(x_test ) 
    if y_test.iloc[0] == 1: # actual 1 
        return np.sum(y_pred)/len(y_pred) , None , np.sum(y_pred)/len(y_pred) 
    
    elif y_test.iloc[0] == 0: # actual 0 
        return 1-np.sum(y_pred)/len(y_pred)  , 1-np.sum(y_pred)/len(y_pred) ,None
    
    
        #Accuracy, TNR, TPR 
        

def correlation_plot(): 
    corr_matrix = df.corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    plt.figure(figsize=(12, 10))    
    sns.heatmap(corr_matrix,mask=mask, annot=False, cmap='coolwarm', fmt=".2f", linewidths=0.5)    
    plt.yticks(rotation=0)
    plt.title('Correlation Coefficient Matrix')
    plt.savefig(dir_ + 'correlationmatrix.pdf', bbox_inches='tight')




def summarize_lambda(X_test, y_test, X_train, y_train, i, df1 ): 
    Cs = [round(i, 2) for i in np.arange(0, 1.05, 0.05)]  # smaller C means stronger regularisation  
    Cs[0] = 0.01 
    
    if df1.empty: 
        df1 = pd.DataFrame(0, index=Cs, columns=['acc_ridge', 'acc_lasso', 'tnr_ridge','tnr_lasso', 'tpr_ridge','tpr_lasso'])
    for C in Cs: 
        #ridge
        model = LogisticRegression(C=C, penalty="l2", solver="liblinear") #l2 = ridge 
        model.fit(X_train, y_train) 
        row = stats_from_model(model, X_test, y_test)    
        df1.at[C , 'acc_ridge'] = df1.at[C , 'acc_ridge'] + row[0] 
        if row[1] == None: 
            df1.at[C, 'tpr_ridge'] = df1.at[C, 'tpr_ridge'] + row[2] 
        else: 
            df1.at[C, 'tnr_ridge'] = df1.at[C, 'tnr_ridge'] + row[1] 
        
        # LASSO 
        model = LogisticRegression(C=C, penalty="l1", solver="liblinear") #l1 = LASSO  
        model.fit(X_train, y_train) 
        row = stats_from_model(model, X_test, y_test)    
        df1.at[C , 'acc_lasso'] = df1.at[C , 'acc_lasso'] + row[0] 
        if row[1] == None: 
            df1.at[C, 'tpr_lasso'] = df1.at[C, 'tpr_lasso'] + row[2] 
        else: 
            df1.at[C, 'tnr_lasso'] = df1.at[C, 'tnr_lasso'] + row[1] 
            
  
    
    return df1 



# =========================== ===========================#
    
def get_lambda(): 
    df1 = pd.DataFrame() 
    unique_name = df.name.unique() 
    
    for i in range(len(unique_name)): 
        poly = PolynomialFeatures(interaction_only=True)
        scaler=StandardScaler() 
        X_test, y_test, X_train, y_train = split_loso(df, unique_name[i] )
        X_train_poly = poly.fit_transform(X_train)
        X_test_poly = poly.transform(X_test)
        
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        X_train_poly = scaler.fit_transform(X_train_poly)
        X_test_poly = scaler.transform(X_test_poly)
        df1 = summarize_lambda(X_test, y_test, X_train , y_train, i, df1 )
    
    df1.acc_ridge = 1- df1.acc_ridge/32
    df1.acc_lasso = 1- df1.acc_lasso /32
    df1.tnr_ridge = df1.tnr_ridge/8
    df1.tnr_lasso = df1.tnr_lasso/8
    df1.tpr_ridge = df1.tpr_ridge/24
    df1.tpr_lasso = df1.tpr_lasso/24
    
    
    
    fig, axes = plt.subplots(2, 1, figsize=(8, 10), sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(df1[df1.columns[0+i]],linewidth=2, color='blue', label='classification error') 
        ax2 = ax.twinx()
        ax2.plot(df1[df1.columns[2+i]],linewidth=2, color='red', label='TNR') # tnr
        ax2.plot(df1[df1.columns[4+i]],linewidth=2, color='green', label='TPR') #tpr 
        if i ==0: 
            ax.set_title('ridge')
            ax.axvline(x=0.65, color='black', linestyle='--') 
            ax.annotate('0.65', xy=(0.65, 0), xycoords='axes fraction', xytext=(0.75, 0.15),
                textcoords='axes fraction', ha='center', va='bottom',
                arrowprops=dict(arrowstyle='->', color='black'))
        else: # lasso 
            ax.set_title('LASSO')
            ax.axvline(x=0.25, color='black', linestyle='--') 
            ax.annotate('0.25', xy=(0.25, 0), xycoords='axes fraction', xytext=(0.35, 0.1),
                textcoords='axes fraction', ha='center', va='bottom',
                arrowprops=dict(arrowstyle='->', color='black'))
            
        ax.set_ylabel('Loss/Rate (%)') 
        ax.set_xlabel('Regularization coefficient')
        ax.legend(loc='right')  
        ax2.legend(loc='upper right')
    plt.tight_layout()  # Adjust layout for better presentation
    #plt.savefig(dir_ + 'selectcoef_interaction.pdf', bbox_inches='tight')  
    
    return df1
        
    

    
    

# =========================== ===========================#
def summarize_components(X_test, y_test, X_train, y_train,i, df1): 
    Ps = list(range(1,18)) # no. of PCs   
    if df1.empty: 
        df1 = pd.DataFrame(0, index=Ps , columns=['acc_pca', 'tnr_pca', 'tpr_pca']) 
    for p in Ps: 
        pca = PCA(n_components = p )
        pc_train = pca.fit_transform(X_train) 
        pc_test = pca.transform(X_test)
        model = LogisticRegression(penalty=None, solver="lbfgs") 
        model.fit(pc_train, y_train)
        row = stats_from_model(model, pc_test, y_test)    
        df1.at[p , 'acc_pca'] = df1.at[p , 'acc_pca'] + row[0] 
        if row[1] == None: 
            df1.at[p, 'tpr_pca'] = df1.at[p, 'tpr_pca'] + row[2] 
        else: 
            df1.at[p, 'tnr_pca'] = df1.at[p, 'tnr_pca'] + row[1] 
        
        
    return df1


def get_components(): # get optimal no. of PCs (p) 
    unique_name = df.name.unique() 
    df1 = pd.DataFrame() 
    
    for i in range(len(unique_name)): 
        poly = PolynomialFeatures(interaction_only=True)
        scaler=StandardScaler() 
        X_test, y_test, X_train, y_train = split_loso(df, unique_name[i] )
        X_train_poly = poly.fit_transform(X_train)
        X_test_poly = poly.transform(X_test)
        
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        X_train_poly = scaler.fit_transform(X_train_poly)
        X_test_poly = scaler.transform(X_test_poly)
        
        
        df1 = summarize_components(X_test_poly, y_test, X_train_poly, y_train,i, df1) 
        
    df1.acc_pca = 1- df1.acc_pca/32
    df1.tnr_pca = df1.tnr_pca/8
    df1.tpr_pca = df1.tpr_pca/24
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 10), sharex=True)

    ax.plot(df1[df1.columns[0]],linewidth=2, color='blue', label='classification error') 
    ax2 = ax.twinx()
    ax2.plot(df1[df1.columns[1]],linewidth=2, color='red', label='TNR') # tnr
    ax2.plot(df1[df1.columns[2]],linewidth=2, color='green', label='TPR') #tpr 
    
    ax.set_title('PCA')
    ax2.axvline(x=6, color='black', linestyle='--') 
  
   
    
        
    ax.set_ylabel('Loss (%)') 
    ax2.set_ylabel('Rate (%)') 
    ax.set_xlabel('Regularization coefficient')
    ax.legend(loc='right')  
    ax2.legend(loc='upper right')
    plt.show() 
    plt.tight_layout()  # Adjust layout for better presentation
    #plt.savefig(dir_ + 'selectp.pdf', bbox_inches='tight')  
    
    return df1




def scree_plot(): 
    X = df.drop('status', axis=1)
    y= df['status']
    X=PolynomialFeatures(interaction_only=True).fit_transform(X)
    X=StandardScaler().fit_transform(X)
    pca=PCA(n_components=18) 
    pca_fit = pca.fit(X)
    PC_values = np.arange(pca.n_components_) + 1
    plt.plot(PC_values, pca.explained_variance_ratio_, 'o-', linewidth=2, color='blue')
    plt.xlabel('No. of PCs')
    plt.xticks(np.arange(min(PC_values), max(PC_values)+1, 1))
    plt.ylabel('Explained Variance')
    for i in range(3): 
        plt.annotate(f"{pca.explained_variance_ratio_[i]*100:.1f}%",
                     xy=(i+1, pca.explained_variance_ratio_[i]), xycoords='data',
                     xytext=(i+1, pca.explained_variance_ratio_[i] + 0.05),
                     textcoords='data', arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
                     fontsize=10, color='black')
    plt.tight_layout()  # Adjust layout for better presentation
    plt.savefig(dir_ + 'screeplot_interaction.pdf', bbox_inches='tight')  # Save the plot as a PDF for LaTeX inclusion

# =========================== ===========================#

   

def log_reg(X_test, y_test, X_train, y_train,X_train_poly, X_test_poly ,iternum, df_pca, df_ridge, df_lasso, df_u ,df2): 
    if df_pca.empty: 
        df_pca = pd.DataFrame(0, index=[] , columns=['accuracy', 'TNR' , 'TPR']) 
        df_ridge, df_lasso, df_u= df_pca.copy() , df_pca.copy() , df_pca.copy() 
        df2 = pd.DataFrame(0, index=list(range(1,155)), columns=['sparsity'])
    #PCA
    pca = PCA(n_components = 5 )
    pc_train = pca.fit_transform(X_train_poly) 
    pc_test = pca.transform(X_test_poly)
    model = LogisticRegression(penalty=None, solver="lbfgs") 
    model.fit(pc_train, y_train)
    row = stats_from_model(model, pc_test, y_test)    
    df_pca.loc[iternum] = row 
    
    # ridge 
    model = LogisticRegression(C=0.10, penalty="l2", solver="liblinear") #l2 = ridge 
    model.fit(X_train, y_train) 
    row = stats_from_model(model, X_test, y_test)    
    df_ridge.loc[iternum] = row 
    
    
    #LASSO 
    model = LogisticRegression(C=0.20, penalty="l1", solver="liblinear") #l1 = LASSO  
    model.fit(X_train_poly, y_train) 
    row = stats_from_model(model, X_test_poly, y_test)    
    df_lasso.loc[iternum] = row 
    
    #LASSO sparsity 
    coefs = model.coef_.ravel() # (17,) array 
    list1.append( np.sum(coefs != 0) )
    df2['sparsity'] = df2['sparsity'] + (coefs != 0 ) 

    
    # standard log reg 
    model = LogisticRegression(penalty=None, solver="lbfgs") 
    model.fit(X_train, y_train) 
    row= stats_from_model(model, X_test, y_test)    
    df_u.loc[iternum] = row 

    return df_pca, df_ridge, df_lasso, df_u ,df2
   
       
    

def run_expt(): 
    unique_name = df.name.unique() # len 8
    df_pca, df_ridge, df_lasso, df_u ,df2 = pd.DataFrame() ,pd.DataFrame(),pd.DataFrame() ,pd.DataFrame() ,pd.DataFrame() 
    
    for i in range(len(unique_name)): 
        poly = PolynomialFeatures(interaction_only=True)
        scaler=StandardScaler() 
        X_test, y_test, X_train, y_train = split_loso(df, unique_name[i] )
        X_train_poly = poly.fit_transform(X_train)
        df2.index = poly.get_feature_names_out(X_train.columns)
        
        X_test_poly = poly.transform(X_test)
        
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        X_train_poly = scaler.fit_transform(X_train_poly)
        X_test_poly = scaler.transform(X_test_poly)
        
        
        df_pca, df_ridge, df_lasso, df_u ,df2= log_reg(X_test, y_test, X_train, y_train,X_train_poly, X_test_poly ,i, df_pca, df_ridge, df_lasso, df_u ,df2)
    return df_pca, df_ridge, df_lasso, df_u ,df2
        

global list1
list1 =[] 
df_pca, df_ridge, df_lasso, df_u ,df2= run_expt() 
# ci = stats.norm.interval(0.95, loc=np.mean(df_pca.accuracy), scale=stats.sem(df_pca.accuracy))
# PCA accuracy 95% CI (0.6910644428231538, 0.9041736524149415) 
# ridge  (0.656333526985111, 0.8957498063482224) 
# LASSO (0.6698292876675882, 0.9135040456657451) 
# standard LR  (0.5971759184639132, 0.8418121767741822)
