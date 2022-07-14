import numpy as np
from numpy import inf
from numpy import nan

import math
from scipy.optimize import differential_evolution
from scipy.stats import entropy

import time

import multiprocessing as mp
from pylab import rcParams

import warnings
warnings.filterwarnings('ignore')


 


#parameters


b2 = -6 # quadratic benefit coefficient for task X
beta =3 # slope of benefit for task Y (eg. brood care)
alpha = 1.5 # selection intensity in social learning
popSize = 500 # population size
n =5 # group size for individual interactions
mu = 0.1 # mutation rate
sigma=0.005 #mutation size
#initial traits
x=0.5 
y=0.5



# Total Benefit Accrue to Individuals Inside a Game
def benefit(G,b1,w):
    
    G=np.array(G)
    X = G[:,0]
    Y = G[:,1]
    total_engagement_in_task_X = np.sum(X)
    total_engagement_in_task_Y = np.sum(Y)
    
    B_X = (b2*(total_engagement_in_task_X**2) + (b1*total_engagement_in_task_X))
    B_Y = (1)/(1+ ((1-w*total_engagement_in_task_Y)/(w*total_engagement_in_task_Y))**beta)
    B_total = (1/n)*(B_X)* B_Y
    return(B_total)


# Cost of Task X
def cost_X(player,b1,w):

     return(-1*(player[0]**2) + 2*(player[0]))



# Cost of Task Y
def cost_Y(player,b1,w):
    
	return(-1*(player[1]**2) + 2*(player[1]))


# Total Cost For an Individual
def cost(player,b1,w):
    return (cost_X(player,b1,w) + cost_Y(player,b1,w))


# Payoff Array of the Individuals Inside a Game
def payoffGame(G,b1,w):

    B = np.array(benefit(G,b1,w))
    C = np.array(list(map(cost,G,[b1]*len(G),[w]*len(G))))
    return (B-C)



# Mutations
def mutate_xy(player):

    
    a = np.random.uniform(0,1) 
    b = np.random.uniform(0,1) 
    if a < mu and b < mu:
        return(both_traits_mutation(player[0],player[1]))
    elif a < mu and b > mu:
        return(only_trait1_mutation(player[0],player[1]))
    elif a > mu and b < mu:
        return(only_trait2_mutation(player[0],player[1]))
    else:
        return(player[0],player[1])

def only_trait1_mutation(trait1,trait2):
    flag =0
    while flag==0:
        temp = np.clip(np.random.normal(trait1,sigma),0,1)
        if temp + trait2 <= 1:
            flag = 1
            return(temp,trait2)
        else:
            continue

def only_trait2_mutation(trait1,trait2):
    flag =0
    while flag==0:
        temp = np.clip(np.random.normal(trait2,sigma),0,1)
        if temp + trait1 <= 1:
            flag = 1
            return(trait1,temp)
        else:
            continue
def both_traits_mutation(trait1,trait2):
    flag =0
    while flag==0:
        temp1 = np.clip(np.random.normal(trait1,sigma),0,1)
        temp2 = np.clip(np.random.normal(trait2,sigma),0,1)
        if temp1 + temp2 <= 1:
            flag = 1
            return(temp1,temp2)
        else:
            continue


# One Round Gameplay in Population
def oneRound(b1,w,alpha):
    global population
    global X
    global Y
    global average_allPayoffs
        
    
    np.random.shuffle(population)
    allGames = np.split(population,popSize/n)
    allPayoff = np.array(list(map(payoffGame, allGames,[b1]*len(allGames),[w]*len(allGames))))
    allPayoffs = allPayoff.flatten()
    average_allPayoffs = sum(allPayoffs)/popSize
    allPayoffs = np.exp(np.multiply(allPayoffs,alpha))
    indexes = np.random.choice(np.arange(0,popSize),popSize,replace=True,p=allPayoffs/np.sum(allPayoffs))
    
    parents = np.array([population[i] for i in indexes])
    a=np.where((abs(population[:,0]-parents[:,0])< 0.2) & (abs(population[:,1]-parents[:,1])< 0.2))
    a=np.ndarray.tolist(a[0])
    population=np.ndarray.tolist(population)
    parents=np.ndarray.tolist(parents)
    for i in a:
        population[i][0]=np.copy(parents[i][0])
        population[i][1]=np.copy(parents[i][1])
    population=np.array(population)
    parents=np.array(parents)

    popTemp = np.array(list(map(mutate_xy,population)))
    X = popTemp[:,0]
    Y = popTemp[:,1]
    population[:,0]= X
    population[:,1]= Y   



# Iterative Game Playing
def play(startTrait, tEnd, plotInterval,b1,w,alpha):
    global popSize
    global population
    global X
    global Y
    global average_allPayoffs
    global all_x_traits
    global all_y_traits
    global all_z_traits
    global all_ids
    all_x_traits=[]
    all_y_traits=[]
    all_z_traits=[]
    all_ids=[]
    init_population = []
    for i in range(popSize):
        init_population=init_population+[[0.5,0.5,i]]
    population = np.array(init_population) 
    
    for t in range(1,tEnd+1):
        if (t%plotInterval == 0):
            all_x_traits.append(X)
            all_y_traits.append(Y)
            all_z_traits.append(1-np.array(X)-np.array(Y))
            all_ids.append(population[:,2])
                
        oneRound(b1,w,alpha)        
