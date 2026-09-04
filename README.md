# Non-asymptotic confidence intervals for neural network predictions

This repository contains the Python implementation of the hybrid method proposed in the paper:

> **"Non-asymptotic confidence intervals for neural network predictions via sequential parameter estimation and conformal prediction"**  
> by Timofeev A.V.

The method combines:
- Non-asymptotic parameter estimation (based on Timofeev's theory) to derive a stopping rule for training.
- Conformal prediction to construct prediction intervals with guaranteed finite-sample coverage.

Two experiments are included:
1. **Synthetic experiment** (`non_asym.py`) – 1D sinusoidal regression.
2. **Real‑data experiment** (`Boston_Housing.py`) – Boston Housing dataset.

## Requirements

- Python 3.6+
- PyTorch
- NumPy
- Matplotlib
- scikit-learn
- (optional) pandas

Install dependencies with:

```bash
pip install torch numpy matplotlib scikit-learn
Usage
Synthetic experiment
Run:

bash
python non_asym.py
This will:

Generate training/validation/test data from y = sin(2πx) + ε, with ε ~ N(0, 0.1²).

Train a neural network (100 hidden units, ReLU) for 500 epochs.

Estimate constants σ²_hat, L0_hat, q_hat from the SGD trajectory.

Compute the theoretical threshold γ(n).

Apply the stopping rule with δ = 0.2 (Scenario 1 – achievable accuracy).

If the stopping time is found, the model at that epoch is used for conformal calibration on the validation set.

Outputs:

experiment_results.csv – epoch‑wise average prediction width and final metrics.

experiment_plot.png – figure with two panels: (left) width evolution and stopping time, (right) predictions and conformal intervals.

To reproduce Scenario 2 (δ = 0.05, too stringent), change desired_pred_error = 0.2 to desired_pred_error = 0.05 in the script.

Boston Housing experiment
Run:

bash
python Boston_Housing.py
This will:

Load the Boston Housing dataset via OpenML.

Split into 60/20/20 train/val/test.

Standardize features.

Train a neural network (50 hidden units, ReLU, weight decay) for 500 epochs.

Estimate constants and compute γ(n).

Apply the stopping rule with δ = 10% of the standard deviation of the validation target, with min_models=50 and warmup_epochs=50.

Outputs:

boston_experiment_results.csv – epoch‑wise width and final metrics.

boston_experiment_plot.png – figure showing width evolution and conformal intervals on test data.

Results
The results presented in the paper are fully reproducible. Expected outputs (based on the corrected version of the paper):

Synthetic (δ = 0.2): stopping time ~160 epochs, test coverage ~98.3%.

Synthetic (δ = 0.05): no stopping (training continues to 500 epochs), coverage ~97.5%.

Boston Housing (δ = 0.1·std): no stopping, coverage ~96.1%.

Note: The stopping rule is heuristic and motivated by theory; rigorous finite‑sample guarantees are provided only by the conformal prediction step (which uses a separate calibration subset, not used for model selection).

License
This project is licensed under the MIT License – see the LICENSE file for details.

Author
Timofeev A.V.
For questions or issues, please use the GitHub issue tracker.