import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap, LocallyLinearEmbedding
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.kernel_approximation import Nystroem
from DSA.dmd import embed_signal_torch
from scipy.signal import convolve


def normalize_dataset(data):
    shape = data.shape
    n_components = shape[-1]
    reshaped_data = data.reshape(-1, n_components)

    mean_vals = np.mean(reshaped_data, axis=0)
    std_vals = np.std(reshaped_data, axis=0)

    # Replace zero std values to avoid division by zero
    std_vals[std_vals == 0] = 1.0

    normalized_data = (reshaped_data - mean_vals) / std_vals

    normalized_data = normalized_data.reshape(shape)
    return normalized_data


def normalize_data(data_list):
    """
    Normalize data across all contexts to have mean=0 and variance=1

    Parameters:
    -----------
    data_list : list of arrays
        List of arrays with shape (n_contexts, n_timepoints, n_components)

    Returns:
    --------
    normalized_data_list : list of arrays
        Normalized data with the same shape as input
    """
    normalized_data_list = []

    for i, region_data in enumerate(data_list):
        normalized_data_list.append(normalize_dataset(region_data))

    return normalized_data_list


def coarse_grain(trajectories, bin_size=5, bins_overlapping=0):
    """
    Bin or sum trajectories over time windows, with optional overlap.

    Parameters
    ----------
    trajectories : np.ndarray
        Shape (conds, steps, dim) or (steps, dim).
    bin_size : int
        Size of each bin.
    bins_overlapping : int
        Number of overlapping steps between bins.

    Returns
    -------
    coarse_trajectories : np.ndarray
        Binned trajectories.
    """
    if trajectories.ndim == 3:
        conds, steps, dim = trajectories.shape
        n_bins = (
            1 + (steps - bin_size) // (bin_size - bins_overlapping)
            if bin_size > bins_overlapping
            else 0
        )
        coarse_trajectories = np.zeros((conds, n_bins, dim))
        for j in range(n_bins):
            start_idx = j * (bin_size - bins_overlapping)
            end_idx = start_idx + bin_size
            coarse_trajectories[:, j] = np.sum(
                trajectories[:, start_idx:end_idx], axis=1
            )
    else:
        steps, dim = trajectories.shape
        n_bins = (
            1 + (steps - bin_size) // (bin_size - bins_overlapping)
            if bin_size > bins_overlapping
            else 0
        )
        coarse_trajectories = np.zeros((n_bins, dim))
        for j in range(n_bins):
            start_idx = j * (bin_size - bins_overlapping)
            end_idx = start_idx + bin_size
            coarse_trajectories[j] = np.sum(trajectories[start_idx:end_idx], axis=0)
    return coarse_trajectories


def pca_reduce(
    data,
    n_components=3,
    flatten_dims=None,
    return_pca=False,
    verbose=False,
    transpose=False,
):
    """
    Apply PCA to data, supporting 2D or 3D input, with flexible flattening.

    Parameters
    ----------
    data : np.ndarray
        Input data. Can be 2D (samples, features) or 3D (trials, time, features).
    n_components : int
        Number of principal components.
    flatten_dims : tuple or None
        If 3D, which dimensions to flatten (default: all but last).
    return_pca : bool
        If True, also return the fitted PCA object.
    verbose : bool
        If True, print explained variance.
    transpose : bool
        If True, transpose data before PCA (for time-major data).

    Returns -------
    reduced : np.ndarray
        PCA-reduced data, reshaped to match input (with last dim = n_components).
    pca : PCA object (optional)
        The fitted PCA object.
    """
    if transpose:
        data = data.T
    orig_shape = data.shape
    if data.ndim == 3:
        if flatten_dims is None:
            # Default: flatten all but last
            data_flat = data.reshape(-1, data.shape[-1])
            unflatten_shape = [orig_shape[0], orig_shape[1], n_components]
        else:
            # Custom flattening
            axes = [orig_shape[i] for i in flatten_dims]
            data_flat = data.reshape(np.prod(axes), orig_shape[-1])
            unflatten_shape = axes + [n_components]
    else:
        data_flat = data
        unflatten_shape = [orig_shape[0], n_components]

    pca = PCA(n_components=n_components)
    reduced_flat = pca.fit_transform(data_flat)
    if isinstance(n_components, float):
        n_components = reduced_flat.shape[-1]
        unflatten_shape[-1] = n_components
    reduced = reduced_flat.reshape(unflatten_shape)
    if verbose:
        print(
            f"PCA explained variance (first {n_components}): {np.cumsum(pca.explained_variance_ratio_)}"
        )
    if return_pca:
        return reduced, pca
    return reduced


def nonlinear_dimensionality_reduction(
    data, method="isomap", n_components=3, n_delays=1, delay_interval=1, **kwargs
):
    """
    Perform nonlinear dimensionality reduction on the input data.

    Parameters:
    -----------
    data : numpy.ndarray
        Input data array. If 3D, the first two dimensions will be flattened.
    method : str
        Dimensionality reduction method to use. Options: 'isomap', 'lle', 'pca'
    n_components : int
        Number of components to reduce to
    **kwargs : dict
        Additional arguments to pass to the specific dimensionality reduction method

    Returns:
    --------
    reduced_data : numpy.ndarray
        Data after dimensionality reduction
    model : object
        The fitted dimensionality reduction model
    """
    if n_delays > 1:
        data = embed_signal_torch(data, n_delays, delay_interval)

    if method in {"None", "id", None, "identity"}:
        return data

    # Reshape data if it's 3D
    original_shape = data.shape
    if len(original_shape) == 3:
        # Flatten the first two dimensions
        data = data.reshape(-1, original_shape[-1])

    # Initialize the appropriate model based on the method string
    if method.lower() == "isomap":
        model = Isomap(n_components=n_components, **kwargs)
    elif method.lower() in {"lle", "locallylinearembedding"}:
        model = LocallyLinearEmbedding(n_components=n_components, **kwargs)
    elif method.lower() == "pca":
        model = PCA(n_components=n_components, **kwargs)
    elif method.lower() == "kernel_pca":
        kernel = kwargs.get("kernel", "rbf")
        if kernel in kwargs:
            kwargs.pop("kernel")
        nystrom_components = kwargs.get("nystrom_components", 100)
        if "nystrom_components" in kwargs:
            kwargs.pop("nystrom_components")

        nystroem = Nystroem(n_components=nystrom_components, kernel=kernel, **kwargs)
        pca = PCA(n_components=n_components)
        model = make_pipeline(nystroem, pca)
    elif method.lower() == "umap":
        #assert that umap is installed
        try:
            from umap import UMAP
        except ImportError:
            raise ImportError("umap is not installed. Please install it with `pip install umap-learn`")
        
        model = UMAP(n_components=n_components, **kwargs)
    else:
        raise ValueError(
            f"Unknown dimensionality reduction method: {method}. "
            f"Supported methods are 'isomap', 'lle', 'pca', and 'kernel_pca'."
        )

    reduced_data = model.fit_transform(data)

    if len(original_shape) == 3:
        reduced_data = reduced_data.reshape(
            original_shape[0], original_shape[1], n_components
        )

    return reduced_data


def featurize_data(
    data, method="id", pca_downsample=False, pca_n_components=3, **kwargs
):
    if data.ndim == 3:
        shape = data.shape
        data = data.reshape(-1, data.shape[-1])
    else:
        shape = data.shape

    if method.lower() == "id":
        pass
    elif method.lower() in {"rbfsampler", "rbf_sampler"}:
        from sklearn.kernel_approximation import RBFSampler

        rbf_sampler = RBFSampler(**kwargs)
        data = rbf_sampler.fit_transform(data)
    elif method.lower() == "nystroem":
        from sklearn.kernel_approximation import Nystroem

        nystroem = Nystroem(**kwargs)
        data = nystroem.fit_transform(data)
    elif method == "polynomial":
        from sklearn.kernel_approximation import PolynomialCountSketch

        data = PolynomialCountSketch(**kwargs).fit_transform(data)
    else:
        raise ValueError(
            f"Unknown featurization method: {method}. Supported methods are 'id' and 'rbfsampler'."
        )

    if pca_downsample:
        data = pca_reduce(data, n_components=pca_n_components)

    if len(shape) == 3:
        return data.reshape(shape[0], shape[1], -1)
    else:
        return data.reshape(shape[0],-1)

def gaussian_filter(data, sigma, truncate=2.0,causal=True,dim=0,mode='same'):
    """
    Applies a causal Gaussian filter to a 1D time series.

    Parameters:
    - data: array-like, the time series data to filter.
    - sigma: float, the standard deviation of the Gaussian kernel.
    - truncate: float, truncate the filter at this many standard deviations.

    Returns:
    - filtered_data: array-like, the smoothed time series.
    """

    kernel_size = int(truncate * sigma + 0.5)
    t = np.arange(-kernel_size, kernel_size + 1)
    kernel = np.exp(-0.5 * (t / sigma)**2)
    kernel /= kernel.sum()

    if causal:
        kernel[t > 0] = 0

    kernel /= kernel.sum()
    kernel = kernel[::-1]

    filtered_data = np.apply_along_axis(
        lambda x: convolve(x, kernel, mode=mode),
        axis=dim,
        arr=data
    )

    return filtered_data, kernel