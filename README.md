# NN_PN

## About

This repository contains code for the paper:

**Neural Network-Based Adaptive Filtering of the Spherical Harmonic Method**  
Benjamin Plumridge, Cory Hauck, Steffen Schotthöfer  
*To appear in the Journal of Scientific Computing, September 2025*  
DOI: [to be added]

---

## Installation

Clone the repo and install dependencies:

```bash
git clone https://github.com/benplumridge/NN_PN.git
cd NN_PN
pip install -r requirements.txt

## 1D Experiments

** Train model for each N in [3,7,9] **

cd 1D
python scripts/train_all.py


** Test all problems **

cd 1D
python scripts/test_all.py


## 2D Experiments

cd 2D
python scripts/train_all.py

##2D Testing

cd 2D
python scripts/test_all.py



