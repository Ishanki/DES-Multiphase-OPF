# DES-Multiphase-OPF
 Optimal design of Distributed Energy Systems (DES) subject to multiphase optimal power flow (MOPF) constraints applicable to distribution networks. A new algorithm with regularised complementarity constraints is used to solve the combined nonconvex problem.

## Technical Information
DES-MOPF refers to the combined DES design model with multiphase AC optimal power flow. 
\
DES-OPF refers to the combined DES design model with balanced AC optimal power flow. 
\
To run a model, use the .py file with "main" in its filename. 
\
The models require one mixed-integer linear solver and nonlinear programming solver.
It is advised to run the models with the MILP solver CPLEX and NLP solver CONOPT (these are the defaults). 
\
Note that the results from the OPF/MOPF classes (voltage magnitudes, angles)
are all returned in p.u. Please use the bases of these to convert them to SI units. 
#### Dependencies and versions used during testing:
Pyomo 5.7.3  \
Pandas 0.25.1 \
Numpy 1.17.0 \
xlrd 1.2.0  

#### Case study:
The original IEEE EU LV Test Case can be found here: 
\
https://cmte.ieee.org/pes-testfeeders/resources/
\
All the input files for the modified test case, which is used to test the models, are provided in each folder. 

## Preprint:
I. De Mel, O. V. Klymenko, and M. Short, “Complementarity Reformulations for the Optimal Design of Distributed Energy Systems with Multiphase Optimal Power Flow,” Apr. 2022, 
\
Available: [link to be added].

## License:
Copyright (c) 2020, Ishanki De Mel. GPL-3.
