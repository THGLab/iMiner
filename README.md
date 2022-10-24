# iMiner
Inhibitor Mining protocol (iMiner) including Consensus Docking and MD Analysis

# Package requirements
* See `requirements.yaml`
* Add newly required packages (including exact versions) to `requirements.yaml`

# Installation
* `pip install -e .`

# Project structure
* `iMiner.core`: Core functions for iMiner
* `iMiner.docking`: Docking protocol base class and implementations using different docking software
* `iMiner.tools`: Useful tools for running iMiner protocol
* `iMiner.md`: GROMACS-based MD simulation module to evaluate ligand binding stability