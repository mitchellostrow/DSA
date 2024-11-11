# DSA
Dynamical Similarity Analysis code accompanying the paper "Beyond Geometry: Comparing the Temporal Structure of Computation in Neural Circuits via Dynamical Similarity Analysis"

https://arxiv.org/abs/2306.10168

How can we tell whether two neural networks are utilizing the same internal processes for a particular computation? This question is pertinent for multiple subfields of both neuroscience and machine learning, including neuroAI, mechanistic interpretability, and brain-machine interfaces. Standard approaches for comparing neural networks focus on the spatial geometry of latent states. Yet in recurrent networks, computations are implemented at the level of neural dynamics, which do not have a simple one-to-one mapping with geometry. To bridge this gap, we introduce a novel similarity metric that compares two systems at the level of their dynamics. Our method incorporates two components: Using recent advances in data-driven dynamical systems theory, we learn a high-dimensional linear system that accurately captures core features of the original nonlinear dynamics. Next, we compare these linear approximations via a novel extension of Procrustes Analysis that accounts for how vector fields change under orthogonal transformation. Via four case studies, we demonstrate that our method effectively identifies and distinguishes dynamic structure in recurrent neural networks (RNNs), whereas geometric methods fall short. We additionally show that our method can distinguish learning rules in an unsupervised manner. Our method therefore opens the door to novel data-driven analyses of the temporal structure of neural computation, and to more rigorous testing of RNNs as models of the brain.

Code Authors: Mitchell Ostrow, Adam Eisen, Leo Kozachkov

If you use this code, please cite:
```
@misc{ostrow2023geometry,
      title={Beyond Geometry: Comparing the Temporal Structure of Computation in Neural Circuits with Dynamical Similarity Analysis}, 
      author={Mitchell Ostrow and Adam Eisen and Leo Kozachkov and Ila Fiete},
      year={2023},
      eprint={2306.10168},
      archivePrefix={arXiv},
      primaryClass={q-bio.NC}
}
```

## Install the repo using `pip`:

```
git clone https://github.com/mitchellostrow/DSA
cd DSA/
pip install -e .
```

## Brief Tutorial

The central object in the package is `DSA`, which links together the `DMD` and `SimilarityTransformDist` (called Procrustes Analysis over Vector Fields in the paper) objects. We designed an API that should be easy to use them in conjunction (`DSA`) with a variety of datatypes for a range of analysis cases:
 * Standard: Comparing two data matrices X, Y (can be passed in as numpy arrays or torch Tensors)
 * Pairwise: Pass in a list of data matrices X, which can be compared all-to-all
 * Disjoint Pairwise: Pass in two lists of data matrices, X, Y, which are compared all-to-all in a bipartite fashion
 * One-to-All: Pass in a list of data matrices X and a single matrix Y. All of X are compared to Y.

# DSA has CUDA capability via pytorch, which is highly recommended for large datasets. 
* Simply pass in `device='cuda'` to the `DSA`,`DMD`,`SimilarityTransformDist` objects to compute on GPU, if you have one. 

Depending on the structure of the data, you can also pass in hyperparameters that vary:
* If your parameters are a single variable, it will be broadcast to all data matrices
* If your parameters are a list of two variables `(a,b)`, each will be broadcast to all data matrices in X and Y, respectively
* If your parameters are a list of two lists `([a,b],[c,d])`, they will be mapped onto to all data matrices in X and Y with corresponding indices. Will throw an error if there aren't enough hyperparamters to match the data.
* If your parameters are a combination of the previous two (e.g. `(a,[b,c])`), the broadcasting behaviors will be combined accordingly.

Our code also uses an API similar to `scikit-learn` in that all the relevant computation is enclosed in the `.fit()`, `.score()`, and `.fit_score()` style functions:
```
dsa = DSA(models,n_delays=n_delays,rank=rank,delay_interval=delay_interval,verbose=True,device=device)
similarities = dsa.fit_score()
```

Simple as that! The data matrices can be of shape `(trials,time,channels)` or `(time,channels)`. If you have multiple conditions you wish to test (for example, different control inputs to your system, you can fit them separately or simultaneously. In our tutorial notebook, `fig3_tutorial.ipynb`, we fit two conditions simultaneously and the model works--here, our data matrices are of shape `(condition,trials,time,channels)` which we collapse to `(condition*trials,time,channels)`.

Note that `DSA` performs multiple fits to the data: one `DMD` matrix per data matrix, and then one `SimilarityTransformDist` similarity per pair of data matrices. When you call `score` after `fit_score`, it will only recompute the `SimilarityTransformDist`s. If you wish to recompute the DMDs, call `.fit_dmds()`. The Procrustes Analysis over Vector Fields metric does not have a closed form solution so it may be worth playing around with its optimization parameters.

If you only care about identifying topological conjugacy between your systems, you can set `compare_method='wasserstein'`, and `wasserstein_compare='eig'` to compare the eigenvalues of the DMDs of each system with the wasserstein distance (as used in Redman et al., 2024 and upcoming work). Optimizing the PAVF metric over O(n) compares transients as well as eigenvalues.

In the case of a large number of comparisons, it will be more memory effective to use the `DMD` class to fit the models and then the `SimilarityTransformDist` class to compare them, rather than use `DSA`, as `DSA` requires taking in all of the data matrices at once. Using the pieces separately will allow you to stream data, or generate it on-the-fly. This process is simple too (see `examples/fig3_tutorial.ipynb`):

* Fit the DMD: with your data:
```
dmd = DMD(x,n_delays=n_delays,rank=rank,delay_interval=delay_interval,device='cuda')
dmd.fit()
Ai = dmd.A_v #extract DMD matrix
```
* Compare with SimilarityTransformDist:
```
comparison = SimilarityTransformDist(device='cuda',iters=2000,lr=1e-3)
score = comparison_dmd.fit_score(Ai,Aj) #fit to two DMD matrices
```

