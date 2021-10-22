#!/bin/bash
conda create -y --name elution python=3.5
conda activate elution

conda install -y -q -c omnia pdbfixer=1.4
conda install -y -q -c conda-forge joblib=0.11
conda install -y -q -c conda-forge six=1.10.0
conda install -y -q -c deepchem mdtraj=1.9.1
conda install -y -q -c conda-forge scikit-learn=0.19.1
conda install -y -q -c conda-forge setuptools=36.2.2
conda install -y -q -c conda-forge networkx=1.11
conda install -y -q -c conda-forge pillow=4.3.0
conda install -y -q -c conda-forge pandas=0.22.0
conda install -y -q -c conda-forge nose=1.3.7 
conda install -y -q -c conda-forge nose-timer=0.7.0
conda install -y -q -c conda-forge flaky=3.3.0
conda install -y -q -c conda-forge zlib=1.2.11
conda install -y -q -c conda-forge requests=2.18.4
conda install -y -q -c conda-forge xgboost=0.6a2
conda install -y -q -c conda-forge simdna=0.4.2
conda install -y -q -c conda-forge jupyter=1.0.0
conda install -y -q -c conda-forge pbr=3.1.1
conda install -y -q -c rdkit rdkit=2017.09.1

conda install -c anaconda cudatoolkit==9.0 cudnn=7.1.2
yes | pip install tensorflow==1.5.0
conda install -y -q -c openeye openeye-toolkits
conda install -y -q -c conda-forge keras=1.2.2
conda install -y -q -c jupyterlab

git clone git@github.com:deepchem/deepchem.git
cd deepchem
git checkout c5f96d7038abb0e87636bc3e4251439b2fe77435
python setup.py install
nosetests -a '!slow' -v deepchem --nologcapture
