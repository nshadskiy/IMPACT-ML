# IMPACT-ML
Intelligent Mass Parametrization for Advanced Classification Tasks with Machine Learning

## Setup

First, the repository is clone via
```bash
git clone --recurse-submodules https://github.com/nshadskiy/IMPACT-ML.git
```

and after going to the folder (`cd IMPACT-ML`) the environment is set up via conda
```bash
conda env create -f environment.yaml
```
don't forget to change the conda environment path in the `environment.yaml` file to your conda installation which is defined as
```
prefix: /work/USER/CONDAPATH/envs/nn_env
```

## Preselection
To run the preselection step, execute the python script and specify the config file:
```bash
python preselection.py --config-file configs/PATH/CONFIG.yaml
```

## Training 
To train a network run:
```bash
python trochscript.py -i /ceph/USER/SOME/PATH/FOLDER -c configs/FILE.yaml -y 2018 -k mt -s even -o SOME_OUTPUT_NAME -e 2000
```