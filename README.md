# HoGRC
Higher-order Granger reservoir computing

This repository contains the code for the paper:
- [(HoGRC) Higher-order Granger reservoir computing: Simultaneously achieving structures inference and accurate prediction of complex dynamics]

In this work, we formulate a new dynamic inference and prediction framework based on reservior computing and Granger causality. 
Our framework can not only accurately infer the higher-order structures of the system, but also significantly outperforms the baseline methods in prediction tasks. 

This respository provides HoGRC methods implemented in PyTorch. And the experimental data can be generated through code.

## Files
- 'main_L63.py' is HGRC experiment for the Lorenz63 system, including data generation, model training and testing.
- 'main_CL63.py' is HGRC experiment for the coupled Lorenz system.
- 'main_rossler.py' is HGRC experiment for the Rossler system.
- 'main_CRo.py' is HGRC experiment for the coupled Rossler system.
- 'main_FHN.py' is HGRC experiment for the coupled FitzHugh-Nagumo system.
- 'main_HR.py' is HGRC experiment for the coupled simplified Hodgkin–Huxley system.
- 'main_L96.py' is HGRC experiment for the Lorenz96 system.
- 'main_NFW.py' is HGRC experiment for the New Four Wings system.
- 'main_colpitts' is HGRC experiment for the Colpitts system.

- 'power_grid' folder contains the experiments for higher-order Kuramoto dynamics on the UK power grid.
- 'draw' folder contains the drawing program and the corresponding pictures in pdf format.

- 'models' folder: used to store model files
- 'dataset' folder: used to store dataset files
- 'results' folder: used to store results files
