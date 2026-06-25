# Hologram-design-conjugate-co-optimization# Leaky Wave Metasurface Example

This folder contains one example optimization script:

```bash
Example_co_design_LCP_RCP.py
```

It designs one phase mask for two circular polarizations and saves the result in `out_dual/`.

## 1. Install Python packages

Use a Python environment with these packages installed:

```bash
pip install jax jaxlib numpy scipy matplotlib optax
```

If you want GPU support for JAX, install the JAX version that matches your CUDA setup.

## 2. Check the input files

The script reads target fields from:

```bash
data/CUNY_logo.mat
data/ASRC_logo.mat
```

Keep these files in the `data/` folder.

## 3. Run the example

From this folder, run:

```bash
python Example_co_design_LCP_RCP.py
```

The default run uses `120000` epochs.


## 4. Check the outputs

Results are saved in:

```bash
out_dual/
```

The folder will contain:

- target intensity and phase images
- designed far-field intensity images
- optimized phase CSV files

During training, the script saves outputs every `150` epochs.
