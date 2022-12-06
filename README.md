# causal_ccm
Package implementing Convergent Cross Mapping for causality inference in dynamical systems as defined by [Sugihara et al (2012)](https://science.sciencemag.org/content/338/6106/496)

### Example usage
For an example how to use, see: https://github.com/PrinceJavier/causal_ccm/blob/main/usage_sample.ipynb 
<br>Source code: https://github.com/PrinceJavier/causal_ccm

### To install
`pip install causal-ccm`

### To use
Say we want to check if X drives Y. We first define `ccm` using:
* `X` and `Y` - time series data
* `tau` - time lag (if `tau=1` we get `[t, t-1, t-2...]` as our shadow manifold embedding
* `E` - embedding dimension (default=2) for the shadow manifold
* `L` - time horizon to consider, defaults at length of time series X

We import the package
<br>`from causal_ccm.causal_ccm import ccm`

We define `ccm`:
<br>`ccm1 = ccm(X, Y, tau, E, L) # define ccm with X, Y time series `

We check the strength of causality measured as correlation in prediction vs true (see Sugihara (2012))
<br>`ccm1.causality()`

We can visualize cross mapping between manifolds of X and Y
<br>`ccm1.visualize_cross_mapping()`

We visualize correlation of X->Y
<br>We stronger correlation = stronger causal relationship
<br>`ccm1.plot_ccm_correls()`

Finally, we can check convergence in predictions (correlations) by computing `ccm1.causality()` for `ccm1` defined with different L values.

### Additional Feature (PAI)

The pai class implements the Pairwise Asymmetric Inference (see McCracken (2014)). The major difference of pai to ccm is the shadow manifold used to predict `X`. To create the manifold, use the `manifold_pattern` and `tau` parameters. For example, `manifold_pattern=[[0, -1, -2],[0]], tau=2` is the same as the shadow manifold `(X_t, X_{t-1*2}, X_{t-2*2}, Y_t)`.

If this package helped you in your work, pls. cite:
```
@software{Javier_causal-ccm_a_Python_2021,
author = {Javier, Prince Joseph Erneszer},
month = {6},
title = {{causal-ccm a Python implementation of Convergent Cross Mapping}},
version = {0.3.3},
year = {2021}
}
```
