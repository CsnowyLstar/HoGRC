# Primary execution process

In the following, we use the coupled Lorenz63 (CL63) system as a concrete example to explain the implementation process of structure inference and dynamics prediction. As described in the file "main\_CL63.py", our code can be divided into five parts:

(1) The first part is \textbf{hyperparameter setting}, where all relevant hyperparameters are stored in the variable "args".
  
(2) The second part is \textbf{data generation}, invoking the file "dataset/Data\_CL.py" to generate experimental data of the CL63 system. And this data is then stored in the ``dataset/data" folder.
  
(3) The third part is \textbf{the configuration of higher-order structures}, which are used as additional inputs for our higher-order RC.
  
(4) The fourth part is the \textbf{model training}. During the model import, we mainly considered three models: RC, PRC, and HoGRC. These models correspond to the files named "Model\_RC.py", "Model\_PRC.py", and "Model\_HoGRC.py", which are located in the "models" directory. It should be noted that we employ the high-order structure from part (3) when calling the HoGRC model and use ridge regression to train the output layer parameters $W_{\text{out}}$ and the bias terms.
  
(5) The last part is the \textbf{model testing}. In testing, we evaluate the one-step and multi-step predictive errors of the updated HoRC model. For the task of structure inference, the one-step predictive error serves as the metric for updating higher-order structures in part (3), where Algorithm 1 is used in the manuscript. After sufficient iterations of the structure inference, the optimal model can accurately execute multi-step dynamics prediction.

