# HoGRC
Higher-order Granger reservoir computing

This repository contains the code for the paper:
- [Higher-order Granger reservoir computing: Simultaneously achieving structures inference and accurate prediction of complex dynamics]

In this work, we formulate a new dynamic inference and prediction framework based on reservior computing and Granger causality. 
Our framework can not only accurately infer the higher-order structures of the system, but also significantly outperforms the baseline methods in prediction tasks. 

This respository provides HoGRC methods implemented in PyTorch. And the experimental data can be generated through code.

## Environment 
To run this project, we will need to set up an environment with Python 3 and install the following Python packages:
- Python 3.9.7
- joblib 1.3.2
- matplotlib 3.8.2
- numpy 1.26.2
- pandas 2.1.4
- scipy 1.11.4
- scikit-learn 1.3.2
- torch 2.1.2
- torchdiffeq 0.2.3

```python
pip install -r requirements.txt
```

## Examples
- 'An_Example_for_Task_1.py' serves as an example for inferring the high-order neighbors of node z in the Loren63 system, and the results are displayed in Fig. 2a in the main text.
```python
python -m An_Example_for_Task_1
```
- 'An_Example_for_Task_2.py' serves as an example for prediction of the coupled FitzHugh-Nagumo system using different methods, and the results are displayed in Figs. 3a and 3c in the main text.
```python
python -m An_Example_for_Task_2
```

![img](https://github.com/CsnowyLstar/HoGRC/blob/main/Simple_example.gif)


## Files
- 'main_L63.py' is HoGRC experiment for the Lorenz63 system, including five parts: hyperparameter setting , data generation, the configuration of higher-order structures, model training and testing.
- 'main_CL63.py' is HoGRC experiment for the coupled Lorenz system.
- 'main_rossler.py' is HoGRC experiment for the Rossler system.
- 'main_CRo.py' is HoGRC experiment for the coupled Rossler system.
- 'main_FHN.py' is HoGRC experiment for the coupled FitzHugh-Nagumo system.
- 'main_HR.py' is HoGRC experiment for the coupled simplified Hodgkin–Huxley system.
- 'main_L96.py' is HoGRC experiment for the Lorenz96 system.
- 'main_NFW.py' is HoGRC experiment for the New Four Wings system.
- 'main_colpitts' is HoGRC experiment for the Colpitts system.

Please refer to the code comments and the "workflow.md" file for detailed execution specifics of these experiments.

## File folders
- 'power_grid' folder contains the experiments for higher-order Kuramoto dynamics on the UK power grid.
- 'models' folder: used to store model files
- 'dataset' folder: used to store dataset files
- 'results' folder: used to store results files
