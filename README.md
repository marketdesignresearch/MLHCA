# Prices, Bids, Values: One ML-Powered Combinatorial Auction to Rule Them All


This is a piece of software used for computing the efficiency of the MLHCA, ML-CCA and the CCA in the spectrum auction test suite (SATS). The proposed MLHCA is described in detail in the following paper:

**Prices, Bids, Values: One ML-Powered Combinatorial Auction to Rule Them All**<br/>
Ermis Soumalias*, Jakob Heiss*, Jakob Weissteiner,  and Sven Seuken.<br/>
*Forthcoming in [ICML 2025 (Spotlight)](https://icml.cc/)*  
Full paper version including appendix: [[pdf](http://arxiv.org/abs/2308.10226)]


The proposed ML-CCA is described in detail in the following paper:

**Machine Learning-powered Combinatorial Clock Auction**<br/>
Ermis Soumalias*, Jakob Weissteiner*, Jakob Heiss, and Sven Seuken.<br/>
*In [Proceedings of the AAAI Conference on Artificial Intelligence Vol 38](https://doi.org/10.1609/aaai.v38i9.28850), Vancouver, CAN, Feb 2024* <br/>
Full paper version including appendix: [[pdf](http://arxiv.org/abs/2308.10226)]

<sub><sup>*These authors contributed equally.</sup></sub>

## Requirements

* Python 3.8
* Java 8 (or later)
  * Java environment variables set as described [here](https://pyjnius.readthedocs.io/en/stable/installation.html#installation)
* JAR-files ready (they should already be)
  * CPLEX (=20.01.0): The file cplex.jar (for 20.01.0) is provided in the folder lib.
  * [SATS](http://spectrumauctions.org/) (=0.8.1): The file sats-0.8.1.jar is provided in the folder lib.
* CPLEX Python API (make sure that your version of CPLEX is compatible with the cplex.jar file in the folder lib).
* Gurobi Python API.

## Dependencies

Prepare your python environment <name_of_your_environment> (whether you do that with `conda`, `virtualenv`, etc.) and activate this environment. Then, install the required packages as provided in the requirements.txt

Using pip:
```bash
$ pip install -r requirements.txt

```

## CPLEX Python API

Install the CPLEX Python API as described [here](https://www.ibm.com/docs/en/icos/20.1.0?topic=cplex-setting-up-python-api).

Concretely, activate your previously created environment <name_of_your_environment>. 

Then locate your cplex home directory and run the file setup.py that is located in <your_CPLEX_home_directory>\IBM\ILOG\CPLEX_Studio201\python (if access is denied, you need to run as admin):

```bash
$ python setup.py install

```

## Gurobi Python API
Using pip:
```bash
$ python -m pip install gurobipy

```


## SATS
In requirements.txt you ran pip install pysats. Finally, you have to set the PYJNIUS_CLASSPATH environment variable to the absolute path of the lib folder.

```bash
conda env config vars set PYJNIUS_CLASSPATH=<path to project>/MLCA_DQ/src/lib
```

When you run conda activate <name_of_your_environment> the environment variable PYJNIUS_CLASSPATH is set to the value you specified above. When you run conda deactivate, this variable is erased.


## How to run

### 1. To start MLHCA for a specific SATS domain (options include GSVM, LSVM, SRVM, and MRVM) and seed and a specific number of Qinit rounds following the CCA price update rule,  run the following command (from inside the src folder):
```bash
python3 sim_mlca_dq_hybrid.py --domain GSVM --qinit 20 --seed 157 
```

### 2. To start ML-CCA for a specific SATS domain (GSVM, LSVM, SRVM, and MRVM) and seed and a specific number of Qinit rounds following the CCA price update rule,  run the following command (from inside the src folder):
```bash
python3 sim_mlca_dq.py --domain GSVM --qinit 20 --seed 157 --new_query_option gd_linear_prices_on_W_v3
```

### 3. To start CCA for a specific SATS domain (GSVM, LSVM, SRVM, and MRVM) and seed and a specific number of Qinit rounds following the CCA price update rule,  run the following command (from inside the src folder):
```bash
python3 sim_mlca_dq.py --domain GSVM --seed 157 --new_query_option cca
```



By changing the dictionary parameters in the sim_mlca_dq_hybrid.py file, one can change various settings of the mechanism, such as whether to use weights and biases tracking, the reserve prices, the mMVNN hyperparameters, and the hyperparameters for next price vector generation. The default values are set to those that we used for our experiments. For all of those hyperparameters in the code, the comments contain a `#NOTE` specifying which hyperparameter in the paper they correspond to. 
The most convenient way of tracking results is WANDB tracking. 
The main plots to look at would be: Efficiency using "Number of Elicited Bids" as a step metric. This plot corresponds to the main efficiency results that we report in the paper. 

## Contact

Maintained by Ermis Soumalias (ErmisCodes), Jakob Weissteiner (weissteiner), and Jakob Heiss (JakobHeiss).



