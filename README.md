# WORKERS-INACTIVITY-IN-SOCIAL-COLONIES (EGT framework)

Surprisingly, the nature's most efficient social organization often have a large proportion of inactive workers in their workforce, sometimes referred to as _lazy workers_. Here, using a simple mathematical model, we show that a parsimonious hypothesis can explain this puzzling social phenomenon. Our model incorporates social interactions and environmental influences into an evolutionary game theoretical (EGT) framework and captures how individuals react to environment by allocating their activity according to environmental conditions. This model shows that inactivity can emerge under specific environmental conditions as a byproduct of the task allocation process.
Our model confirms that in the case of worker loss, prior homeostatic balance is re-established by replacing some of the lost force with previously inactive workers. Most importantly, our model shows that inactivity in social colonies can be explained without the need to assume an adaptive function for this phenomenon. 

"**Explaining workers' inactivity in social colonies from first principles**" 

Moein Khajehnejad, Julian García, Bernd Meyer

## Model Structure 
#### Schematic diagram of different steps in the social learning paradigm from an EGT-based perspective. 

<img width="600" alt="![schematic]" src="https://user-images.githubusercontent.com/22978025/173978596-b93a95d5-1ce9-4fee-80b7-4534c7ecba1c.png">

## Python Requirements
numpy<br />
math<br />
from scipy.optimize import differential_evolution<br />
from scipy.stats import entropy<br />
time<br />
multiprocessing<br />
from pylab import rcParams
 

## Usage
- Load the required python packages
- Run the jupyter notebook including the following steps:
    - Simulating iterative games in the environment
    - Plotting the dynamics of population traits in time
    - Calculating and plotting the efficiency of the system in the parameter space using stochastic global optimisation meta-heuristic, Differential Evolution

**Inputs:** The model receives the following as the input information in the "_Run_" function of the script:
- _startTrait_: A list of lists (_[[x,y]]_) defining the intitial trait values of the population for tasks X and Y.
- _tEnd_: An integer defining the end time of the simulations in steps.
- _plotInterval_: An integer to determine the distance between time points to render the scatterplots.
- $b_1$: A float number representing linear benefit coefficient for task _X_ (eg. thermoregulation).
- $w$: A float number representing inflection point of benefit for task _Y_ (eg. broodcare).
- $\alpha$ : A float number representing selection intensity in social learning.

**Outputs:** The output would be:
-  Scatterplots illustrating the evolution of population traits in time
-  A heatmap representing the systme's efficiency in the ($w$,$b_1$ )  parameter space.

## Cite Our Work
If you find this model useful to your work, please also cite our paper.
