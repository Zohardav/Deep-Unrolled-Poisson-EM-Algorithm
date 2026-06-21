# Regularized Poisson MLEM Unfolding Playground

This workspace contains a Colab-compatible notebook:

- `poisson_regularized_mlem_unfolding_playground.ipynb`

The notebook builds a synthetic Poisson inverse problem with deterministic background, runs classical regularized MLEM, and compares unfolded MLEM variants. It supports matched and mismatched reconstruction settings, configurable supervised training loss, layer-wise unfolded training, gamma sweeps, and reconstruction/metric plots.

## How To Run

1. Open Google Colab or Jupyter.
2. Open `poisson_regularized_mlem_unfolding_playground.ipynb`.
3. In Colab, select `Runtime > Change runtime type > GPU` if a compatible GPU is available.
4. Edit the `Config` cell if needed.
5. Run cells from top to bottom.

The notebook currently defaults to the full playground settings:

```python
quick_run: bool = False
```

For a faster smoke test, set:

```python
quick_run: bool = True
```

## Main Controls

Most settings are in the `Config` dataclass near the top of the notebook.

### Model Selection

```python
EXPERIMENT_NAME = "non_mismatched"  # or "mismatched"
MODEL_NAME = "denominator_correction"
RUN_ALL_MODELS_IN_EXPERIMENT = True
```

Supported model names:

- `classical_regularized`
- `learned_gamma`
- `denominator_correction`
- `learned_objective`

When `RUN_ALL_MODELS_IN_EXPERIMENT=True`, the notebook evaluates all relevant models for the selected experiment. The mismatched experiment includes the learned-objective model.

### Training Loss

Choose the supervised train/validation loss with:

```python
reconstruction_loss: str = "mse"
```

Supported values:

- `mse`: mean squared error over all pixels.
- `frobenius`: mean per-sample Frobenius norm of the reconstruction error.

Aliases such as `fro`, `frobenious`, and `frobinious` are normalized to `frobenius`.

### Training Schedule

The unfolded models use an interleaved layer-wise schedule:

1. Sequential pass at layer `k`: only the current layer is enabled.
2. End-to-end pass after layer `k > 0`: all layers up to `k` are enabled.

The main controls are:

```python
sequential_train: bool = True
sequential_epochs_per_stage: int = 100
end_to_end_epochs: int = 250
learning_rate: float = 1e-3
end_to_end_learning_rate: float = 5e-3
```

If `sequential_train=False`, trainable unfolded models use a full-depth end-to-end fallback.

## Experiments

The notebook contains two experiment blocks:

- Non-mismatched: data generation and reconstruction both use `A_true, b_true`.
- Mismatched: data are generated from `A_true, b_true`, while reconstruction uses perturbed nominal parameters.

Mismatch behavior is controlled by:

```python
A_mismatch_mode: str = "multiplicative_scale"
A_mismatch_scale_std: float = 0.25
A_mismatch_drop_probability: float = 0.7
b_mismatch_mode: str = "scale_and_offset"
b_nominal_scale: float = 0.7
b_nominal_offset: float = 0.1
```

## Gamma Sweep Study

Before the main experiments, the notebook includes a classical regularized MLEM gamma sweep:

```python
run_gamma_sweep_study = True
gamma_sweep_values = np.array([0.0, 3e-3, 3e-2, 3e-1], dtype=np.float32)
gamma_sweep_num_runs = 50
gamma_sweep_num_iters = config.num_mlem_iters
```

For each gamma value, metrics are averaged over `gamma_sweep_num_runs` independent MLEM runs. The study produces:

- A 2x3 metric plot: MSE, SSIM, and PSNR versus iteration for non-mismatched and mismatched cases.
- Reconstruction image grids for both cases, with `len(gamma_sweep_values)` rows and 5 selected EM iteration columns.

## Outputs

The notebook reports and visualizes:

- Number of learnable parameters per method.
- Training and validation loss curves using the selected training loss.
- MSE, SSIM, and PSNR versus iteration or unfolded layer.
- Final comparison tables with reconstruction and objective diagnostics.
- Reconstruction image grids using a `jet` colormap where exact zero-valued pixels are black.
- Learned gamma values, denominator correction diagnostics, and learned-objective parameter diagnostics when applicable.

## Architecture Notes

- Classical MLEM has zero learnable parameters.
- `learned_gamma` learns one regularization gamma per unfolded layer.
- `denominator_correction` learns one gamma and one correction CNN per unfolded layer.
- `learned_objective` learns per-layer gamma values and shared effective objective parameters `A_eff` and `b_eff`.

Checkpoints are saved only when:

```python
save_checkpoints: bool = True
```

Saved models are written under `trained_models/`.
