# DSA
Dynamical Similarity Analysis code accompanying the paper "Beyond Geometry: Comparing the Temporal Structure of Computation in Neural Circuits via Dynamical Similarity Analysis"

https://arxiv.org/abs/2306.10168

How can we tell whether two neural networks are utilizing the same internal processes for a particular computation? This question is pertinent for multiple subfields of both neuroscience and machine learning, including neuroAI, mechanistic interpretability, and brain-machine interfaces. Standard approaches for comparing neural networks focus on the spatial geometry of latent states. Yet in recurrent networks, computations are implemented at the level of neural dynamics, which do not have a simple one-to-one mapping with geometry. To bridge this gap, we introduce a novel similarity metric that compares two systems at the level of their dynamics. Our method incorporates two components: Using recent advances in data-driven dynamical systems theory, we learn a high-dimensional linear system that accurately captures core features of the original nonlinear dynamics. Next, we compare these linear approximations via a novel extension of Procrustes Analysis that accounts for how vector fields change under orthogonal transformation. Via four case studies, we demonstrate that our method effectively identifies and distinguishes dynamic structure in recurrent neural networks (RNNs), whereas geometric methods fall short. We additionally show that our method can distinguish learning rules in an unsupervised manner. Our method therefore opens the door to novel data-driven analyses of the temporal structure of neural computation, and to more rigorous testing of RNNs as models of the brain.

Code Authors: Mitchell Ostrow, Adam Eisen, Leo Kozachkov

