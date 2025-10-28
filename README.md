# generalized DSA
Computational techniques for Dynamical Similarity Analysis. First introduced in,

1. "Beyond Geometry: Comparing the Temporal Structure of Computation in Neural Circuits via Dynamical Similarity Analysis"

https://arxiv.org/abs/2306.10168

Abstract:
How can we tell whether two neural networks are utilizing the same internal processes for a particular computation? This question is pertinent for multiple subfields of both neuroscience and machine learning, including neuroAI, mechanistic interpretability, and brain-machine interfaces. Standard approaches for comparing neural networks focus on the spatial geometry of latent states. Yet in recurrent networks, computations are implemented at the level of neural dynamics, which do not have a simple one-to-one mapping with geometry. To bridge this gap, we introduce a novel similarity metric that compares two systems at the level of their dynamics. Our method incorporates two components: Using recent advances in data-driven dynamical systems theory, we learn a high-dimensional linear system that accurately captures core features of the original nonlinear dynamics. Next, we compare these linear approximations via a novel extension of Procrustes Analysis that accounts for how vector fields change under orthogonal transformation. Via four case studies, we demonstrate that our method effectively identifies and distinguishes dynamic structure in recurrent neural networks (RNNs), whereas geometric methods fall short. We additionally show that our method can distinguish learning rules in an unsupervised manner. Our method therefore opens the door to novel data-driven analyses of the temporal structure of neural computation, and to more rigorous testing of RNNs as models of the brain.

and now including code from the following: 

2. "InputDSA: Demixing then comparing recurrent and externally driven dynamics
Abstract:
In control problems and basic scientific modeling, it is important to compare observations with dynamical simulations. For example, comparing two neural systems can shed light on the nature of emergent computations in the brain and deep neural networks. Recently, (Ostrow et al., 2023) introduced Dynamical Similarity Analysis (DSA), a method to measure the similarity of two systems based on their state dynamics rather than geometry or topology. However, DSA does not consider how inputs affect the dynamics, meaning that two similar systems, if driven differently, may be classified as different. Because real-world dynamical systems are rarely autonomous, it is important to account for the effects of input drive. To this end, we introduce a novel metric for comparing both intrinsic (recurrent) and input-driven dynamics, called InputDSA (iDSA). InputDSA extends the DSA framework by estimating and comparing both input and intrinsic dynamic operators using a novel variant of Dynamic Mode Decomposition with control (DMDc) based on subspace identification. We demonstrate that InputDSA can successfully compare partially observed, input-driven systems from noisy data. We show that when the true inputs are unknown, surrogate inputs can be substituted without a major deterioration in similarity estimates. We apply InputDSA on Recurrent Neural Networks (RNNs) trained with Deep Reinforcement Learning, identifying that high-performing networks are dynamically similar to one another, while low-performing networks are more diverse. Lastly, we apply InputDSA to neural data recorded from rats performing a cognitive task, demonstrating that it identifies a transition from input-driven evidence accumulation to intrinsically- driven decision-making. Our work demonstrates that InputDSA is a robust and efficient method for comparing intrinsic dynamics and the effect of external input
on dynamical systems

Code Authors: Mitchell Ostrow, Adam Eisen, Leo Kozachkov, Ann Huang

If you use this code, please cite:
```
@misc{huangostrow2025input,
      title={InputDSA: Demixing then comparing recurrent and externally driven dynamics}, 
      author={Ann Huang and Mitchell Ostrow and Satpreet Singh and Leo Kozachkov and Ila Fiete and Kanka Rajan},
      year={2025},
      archivePrefix={arXiv},
      primaryClass={q-bio.NC}
}

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
pip install dsa-metric
```
or if you wish to install the most recent version:
```
git clone https://github.com/mitchellostrow/DSA
cd DSA/
pip install -e .
```

## Brief Tutorial

The central object in the package is `GeneralizedDSA`, which links together the different types of `DMD` and `SimilarityTransformDist` (called Procrustes Analysis over Vector Fields in the first paper) objects. We designed an API that should be easy to use them in conjunction (`DSA`) with a variety of datatypes for a range of analysis cases:
 * Standard: Comparing two data matrices X, Y (can be passed in as numpy arrays or torch Tensors)
 * Pairwise: Pass in a list of data matrices X, which can be compared all-to-all
 * Disjoint Pairwise: Pass in two lists of data matrices, X, Y, which are compared all-to-all in a bipartite fashion
 * One-to-All: Pass in a list of data matrices X and a single matrix Y. All of X are compared to Y.

To run the DSA algorithm as it is specified in Ostrow et al. (2023), the class `DSA` in the file `dsa.py` is recommended. This is a restriction / special case of Generalized DSA. To run the InputDSA algorithm as it is specified in Huang and Ostrow et al. (2025), the class `InputDSA` is recommended. 

The `GeneralizedDSA` class generalizes (hence the name) the `DSA` algorithm from Ostrow et al. (2023) to account for the fact that other types of embeddings and DMD models can improve on HAVOK/Hankel DMD (which applies standard ridge least-squares regression on whitened time-delay embeddings). To that end, we have integrated capabilities with PyKoopman (https://github.com/dynamicslab/pykoopman) and PyDMD (https://github.com/PyDMD/PyDMD) to allow for other DMD models. For a brief tutorial, see below. Likewise, other similarity metrics (e.g. Huang and Ostrow et al., 2025) are desirable as well, given the setting. For kernel-like embeddings, functions in the file `preprocessing.py` can be applied. 

# DSA has CUDA capability via pytorch, which is highly recommended for large datasets. 
* Simply pass in `device='cuda'` to the `DSA`,`DMD`,`SimilarityTransformDist` objects to compute on GPU, if you have one. 

Depending on the structure of the data, you can also pass in hyperparameters that vary:
* If your parameters are a single variable, it will be broadcast to all data matrices
* If your parameters are a list of two variables `(a,b)`, each will be broadcast to all data matrices in X and Y, respectively
* If your parameters are a list of two lists `([a,b],[c,d])`, they will be mapped onto to all data matrices in X and Y with corresponding indices. Will throw an error if there aren't enough hyperparamters to match the data.
* If your parameters are a combination of the previous two (e.g. `(a,[b,c])`), the broadcasting behaviors will be combined accordingly.


Our code also uses an API similar to `scikit-learn` in that all the relevant computation is enclosed in the `.fit()`, `.score()`, and `.fit_score()` style functions. The original DSA case can be applied as follows:
```
from DSA import DSA
dsa = DSA(models,n_delays=n_delays,rank=rank,delay_interval=delay_interval,verbose=True,device=device,score_method='angular')
similarities = dsa.fit_score()
```
If you wish to use pykoopman/pydmd DMD models, they can applied as follows, using the pk.Koopman wrapper class. We'll use the pydmd SubspaceDMD as an example:
```
from DSA import DSA
from pydmd import SubspaceDMD #the DMD class you want to use
import DSA.pykoopman as pk 
obs = pk.observables.TimeDelay(n_delays=3) #define some nonlinear observables, if you wish

dsa = DSA(compare_dat,dmd_class=pk.Koopman,score_method='wasserstein',wasserstein_compare='eig',observables=obs,regressor=SubspaceDMD(svd_rank=3))
```

Due to the generalization of the method on different DMDs and different similarity metrics, each which have different arguments, we have changed the structure of the DSA class to take in arguments for each of these objects as dictionaries or dataclass config objects. Here are a few examples:
```
from dataclass import dataclass
@dataclass()
class DefaultDMDConfig:
    n_delays: int = 1
    delay_interval: int = 1
    rank: int = None
    lamb: float = 0
    send_to_cpu: bool = False
@dataclass()
class pyKoopmanDMDConfig:
    observables = pykoopman.observables.TimeDelay(n_delays=1)
    regressor = pydmd.DMD(svd_rank=2)
    
@dataclass()
class SubspaceDMDcConfig:
    n_delays: int = 1
    delay_interval: int = 1
    rank: int = None
    lamb: float = 0
    backend: str = 'n4sid'

#__Example config dataclasses for similarity transform distance #
@dataclass
class SimilarityTransformDistConfig:
    iters: int = 1500
    score_method: Literal["angular", "euclidean"] = "angular"
    lr: float = 5e-3
    zero_pad: bool = False
    wasserstein_compare: Literal["sv", "eig", None] = "eig"

@dataclass()
class ControllabilitySimilarityTransformDistConfig:
    score_method: Literal["euclidean", "angular"] = "euclidean"
    compare = 'state'
    joint_optim: bool = False
    return_distance_components: bool = False

```
Then, these are passed directly into the GeneralizedDSA (DSA, InputDSA) classes via the arguments dmd_config, simdist_config for the arguments of each class:
```
from DSA import GeneralizedDSA, DMD, SimilarityTransformDist
gdsa = GeneralizedDSA(datasets,dmd_class=DMD,similarity_class=SimilarityTransformDist,
      dmd_config=DefaultDMDConfig,simdist_config=SimilarityTransformDistConfig)
sim = gdsa.fit_score()
```

The logic for InputDSA is equivalent, with a few key things to note. In this setting, there are two types of DMDc models to use-- DMDc (Proctor et al., 2016), and SubspaceDMDc (Huang and Ostrow et al., 2025). If your system is partially observed, we recommend SubspaceDMDc instead. Likewise, there are a few different types of similarity that can be computed. You may wish to apply DMDc-like models but then only compare the A matrix-- in this case, you can set the argument `compare='state'` in the simdist_config object. Otherwise, you have the options `joint,control` which will jointly compare A and B via controllability, or just the control matrix via Procrustes. InputDSA has one other special argument: `return_distance_components`. If this is true, it will return 3 different metrics, encoded in a single numpy array (data x data x 3). They have the ordering: Full Controllability distance, Jointly optimized State Similarity Score, Jointly Optimized Control Score. 


Simple as that! The data matrices can be of shape `(trials,time,channels)` or `(time,channels)`. If you have multiple conditions you wish to test (for example, if you have different task settings in your system, you can fit them separately or simultaneously). In our tutorial notebook, `dsa_fig3_tutorial.ipynb`, we fit two conditions simultaneously and the model works--here, our data matrices are of shape `(condition,trials,time,channels)` which we collapse to `(condition*trials,time,channels)`. (As of 2025, DSA objects can also take lists of arrays (shape 2D or 3D) to account for different lengths of time in different time series). 

Note that `DSA` performs multiple fits to the data: one `DMD` matrix per data matrix, and then one `SimilarityTransformDist` similarity per pair of data matrices. When you call `score` after `fit_score`, it will only recompute the `SimilarityTransformDist`s. If you wish to recompute the DMDs, call `.fit_dmds()`. The Procrustes Analysis over Vector Fields metric does not have a closed form solution so it may be worth playing around with its optimization parameters, or use the Wasserstein distance.

If you only care about identifying topological conjugacy between your systems, you can set `compare_method='wasserstein'`, and `wasserstein_compare='eig'` to compare the eigenvalues of the DMDs of each system with the wasserstein distance (as used in Redman et al., 2024 and upcoming work). Optimizing the PAVF metric over O(n) compares transients as well as eigenvalues.

In the case of a large number of comparisons, it will be more memory effective to use the `DMD` class to fit the models and then the `SimilarityTransformDist` class to compare them, rather than use `DSA`, as `DSA` requires taking in all of the data matrices at once. Using the pieces separately will allow you to stream data, or generate it on-the-fly. This process is simple too (see `examples/dsa_fig3_tutorial.ipynb`):

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
This pipeline can also be generalized using different DMDs and comparison methods. 
