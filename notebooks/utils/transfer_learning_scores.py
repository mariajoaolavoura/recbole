import numpy as np 
import itertools


# ACC, BWT, e FWT - Lopez-Paz e Ranzato GEM

def avg_recall(results_matrix): # Lopez-Paz e Ranzato GEM 2017
    return np.mean( np.diag(results_matrix) )

def compute_BWT_lopes_ranzato(results_matrix): # Lopez-Paz e Ranzato GEM 2017
    BWT = []
    n_checkpoints = results_matrix.shape[0]
    for T in range(1, n_checkpoints): # 1 means holdout 2, 2 means 3, so on
        Rti = results_matrix.iloc[T, 0:T] # get models performances' on previous holdouts
        Rii = np.diag(results_matrix)[0:T] # get models performances' on their closest holdouts (diagonal)
        E = sum( Rti - Rii ) # future models performances' - performances' of models closest to holdouts (diagonal)
        BWT.append( E/T ) # store average BWT for model
    return BWT, np.mean( BWT ) # return BWT and average BWT for all models

def compute_BWT_rodrigues(results_matrix): # Díaz-Rodriguez et al. 2018
    diff = []
    n_checkpoints = results_matrix.shape[0]
    for i in range(1, n_checkpoints): # 1 means holdout 2, 2 means 3, so on
        for j in range(i):
            Rij = results_matrix.iloc[i,j] # get models performances' on previous holdouts
            Rjj = results_matrix.iloc[j,j] # get models performances' on their closest holdouts (diagonal)
            # print(Rij, Rjj)
            diff.append( Rij - Rjj ) # future models performances' - performances' of models closest to holdouts (diagonal)
            # print(diff)
    BWT = sum(diff) / ( n_checkpoints*(n_checkpoints-1) / 2 ) # store average BWT for model
    return BWT, diff # return BWT and average BWT for all models

def compute_FWT_rodrigues(results_matrix): # Díaz-Rodriguez et al. 2018
    upper_tri = results_matrix.to_numpy()[np.triu_indices(results_matrix.shape[0], k=1)]
    return np.mean(upper_tri)


def compute_symmetric_BWT_rodrigues(results_matrix): # Díaz-Rodriguez et al. 2018
    diff = []
    n_checkpoints = results_matrix.shape[0]
    for i in range(0, n_checkpoints-1): # 1 means holdout 2, 2 means 3, so on
        for j in range(i+1, n_checkpoints):
            Rij = results_matrix.iloc[i,j] # get models performances' on previous holdouts
            Rjj = results_matrix.iloc[j,j] # get models performances' on their closest holdouts (diagonal)
            # print(Rij, Rjj)
            diff.append( Rij - Rjj ) # future models performances' - performances' of models closest to holdouts (diagonal)
            # print(diff)
    BWT_symmetric = sum(diff) / ( n_checkpoints*(n_checkpoints-1) / 2 ) # store average BWT for model
    return BWT_symmetric, diff # return BWT and average BWT for all models




def compute_symmetric_BWT_rodrigues_three_main_diagonals(results_matrix): # Díaz-Rodriguez et al. 2018
    diff = []
    n_checkpoints = results_matrix.shape[0]
    for j in range(1, n_checkpoints):
        i = j-1
        # print(i,j)
        Rij = results_matrix.iloc[i,j] # get models performances' on previous holdouts
        Rjj = results_matrix.iloc[j,j] # get models performances' on their closest holdouts (diagonal)
        # print(Rij, Rjj)
        diff.append( Rij - Rjj ) # future models performances' - performances' of models closest to holdouts (diagonal)
        # print(diff)
    # print(diff)
    BWT_symmetric = sum(diff) / ( n_checkpoints*(n_checkpoints-1) / 2 ) # store average BWT for model
    return BWT_symmetric, diff # return BWT and average BWT for all models


def compute_BWT_rodrigues_three_main_diagonals(results_matrix): # Díaz-Rodriguez et al. 2018
    diff = []
    n_checkpoints = results_matrix.shape[0]
    for j in range(0, n_checkpoints-1):
        i = j+1
        # print(i,j)
        Rij = results_matrix.iloc[i,j] # get models performances' on previous holdouts
        Rjj = results_matrix.iloc[j,j] # get models performances' on their closest holdouts (diagonal)
        # print(Rij, Rjj)
        diff.append( Rij - Rjj ) # future models performances' - performances' of models closest to holdouts (diagonal)
        # print(diff)
    # print(diff)
    BWT = sum(diff) / ( n_checkpoints*(n_checkpoints-1) / 2 ) # store average BWT for model
    return BWT, diff # return BWT and average BWT for all models