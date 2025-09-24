# NN_FPN

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

## Test all problems

cd 2D
python scripts/test_all.py
```
### NOTE: If you would like to test problems with already trained models, you can skip training and test with models used in the paper.   The test scripts use models from the `trained_models\` folder.  Training overwrites these models. 

### Individual Experiments

In addition to the "all-in-one" scripts (`train_all.py` and `test_all.py`), individual experiments can be performed using `scripts/test_driver`.  To modify parameters, including N, see `src/parameters`.:

This allows users to:
- Run a single experiment for quick testing.
- Modify individual configurations without running the full suite.
- Reproduce specific figures or results from the paper.

**Recommendation:** For most purposes, use `train_all.py` / `test_all.py` to reproduce all results.




