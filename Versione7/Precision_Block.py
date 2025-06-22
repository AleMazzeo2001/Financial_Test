import random
import numpy as np
import matplotlib.pyplot as plt
import financial_test as FCA

def generate_block_clustered_matrix(n, clusters, overlap_size=0, std_devs=None):
    """
    Generate a block-clustered matrix with overlapping clusters and random correlations outside blocks.

    Parameters:
    - n: Total number of variables.
    - clusters: List of cluster sizes (e.g., [10, 15, 20] means 3 clusters of those sizes).
    - overlap_size: Number of overlapping variables between consecutive clusters (can be a single integer or a list).
    - std_devs: List of standard deviations for each variable (optional).

    Returns:
    - block_matrix: The generated block-structured matrix (n x n).
    """
    if isinstance(overlap_size, list):
        assert len(overlap_size) == len(clusters) - 1, \
            "When overlap_size is a list, its length must be one less than the number of clusters."
    else:
        overlap_size = [overlap_size] * (len(clusters) - 1)

    non_overlap_total = sum(clusters) - sum(overlap_size)
    assert non_overlap_total == n, \
        f"Cluster sizes with overlap don't match total variables. Expected {n}, but got {non_overlap_total}."

    block_matrix = np.zeros((n, n))
    start_idx = 0

    for i, cluster_size in enumerate(clusters):
        end_idx = start_idx + cluster_size

        block = np.random.uniform(0.5, 1, size=(cluster_size, cluster_size))
        block = (block + block.T) / 2
        np.fill_diagonal(block, 1)

        block_matrix[start_idx:end_idx, start_idx:end_idx] = block

        if i < len(overlap_size) and overlap_size[i] > 0:
            current_overlap = overlap_size[i]
            if current_overlap > 0:
                overlap_end_idx = start_idx + cluster_size
                overlap_block = np.random.uniform(0.5, 1, size=(current_overlap, current_overlap))
                overlap_block = (overlap_block + overlap_block.T) / 2
                np.fill_diagonal(overlap_block, 1)

                block_matrix[overlap_end_idx - current_overlap:overlap_end_idx, 
                             overlap_end_idx - current_overlap:overlap_end_idx] = overlap_block

        start_idx = end_idx - (overlap_size[i] if i < len(overlap_size) else 0)

    block_matrix = (block_matrix + block_matrix.T) / 2

    if std_devs is not None:
        std_devs = np.array(std_devs)
        assert len(std_devs) == n, "The number of standard deviations must match the number of variables."
        block_matrix = block_matrix * np.outer(std_devs, std_devs)

    eigvals = np.linalg.eigvalsh(block_matrix)
    if np.any(eigvals < 0):
        min_eig = np.min(eigvals)
        block_matrix += np.eye(n) * (-min_eig + 0.01)

    return block_matrix


def generate_random_block_matrix(min_cluster_size=2, max_cluster_size=5, min_clusters=1, max_clusters=5):
    """
    Generate a symmetric block-structured matrix with randomly sized and positioned blocks with no overlap.

    Parameters:
    - min_cluster_size (int): Minimum size of a cluster.
    - max_cluster_size (int): Maximum size of a cluster.
    - max_clusters (int): Maximum number of clusters.

    Returns:
    - np.ndarray: Symmetric block-structured matrix.
    """
    num_clusters = random.randint(min_clusters, max_clusters)
    clusters = [random.randint(min_cluster_size, max_cluster_size) for _ in range(num_clusters)]

    total_cluster_size = sum(clusters) + (num_clusters - 1) * max_cluster_size
    matrix_size = total_cluster_size

    block_matrix = np.zeros((matrix_size, matrix_size))
    used_positions = []

    for cluster_size in clusters:
        placed = False
        attempts = 0
        while not placed and attempts < 100:
            start_row = random.randint(0, matrix_size - cluster_size)
            start_col = random.randint(0, matrix_size - cluster_size)

            conflict = False
            for (existing_start_row, existing_end_row, existing_start_col, existing_end_col) in used_positions:
                if not (end_row <= existing_start_row or start_row >= existing_end_row or
                        end_col <= existing_start_col or start_col >= existing_end_col):
                    conflict = True
                    break

            if conflict:
                attempts += 1
                continue

            end_row = start_row + cluster_size
            end_col = start_col + cluster_size

            block = np.random.randn(cluster_size, cluster_size)
            block = (block + block.T) / 2
            block /= np.max(np.abs(block))
            block += 1
            np.fill_diagonal(block, 1.0)

            block_matrix[start_row:end_row, start_col:end_col] = block
            block_matrix[start_col:end_col, start_row:end_row] = block

            used_positions.append((start_row, end_row, start_col, end_col))
            placed = True

    block_matrix = make_positive_semidefinite(block_matrix)

    return block_matrix, clusters


def is_positive_semidefinite(matrix):
    """
    Check if a matrix is positive semi-definite by examining its eigenvalues.

    Parameters:
    - matrix (np.ndarray): Matrix to check.

    Returns:
    - bool: True if the matrix is positive semi-definite, False otherwise.
    """
    eigenvalues = np.linalg.eigvalsh(matrix)
    return np.all(eigenvalues >= 0)


def make_positive_semidefinite(matrix, epsilon=1e-6):
    """
    Ensure the matrix is positive semi-definite by adjusting eigenvalues if needed.

    Parameters:
    - matrix (np.ndarray): Matrix to adjust.
    - epsilon (float): Small value added to the diagonal to enforce positive semi-definiteness if needed.

    Returns:
    - np.ndarray: Positive semi-definite matrix.
    """
    if is_positive_semidefinite(matrix):
        return matrix

    identity_matrix = np.eye(matrix.shape[0])
    while not is_positive_semidefinite(matrix):
        matrix += epsilon * identity_matrix
        epsilon *= 10

    return matrix


def generate_matrix(cluster='random', 
                    cluster_size=[4, 10],
                    min_num_cluster=1,
                    max_num_cluster=8,
                    overlap=None, 
                    max_overlap=3, 
                    matrix_type="diagonal"):
    """
    Generate a random block-structured matrix with specified parameters.

    Parameters:
    - cluster: Cluster sizes ('random' for random sizes or specific list).
    - cluster_size: Range of cluster sizes for random generation.
    - min_num_cluster: Minimum number of clusters.
    - max_num_cluster: Maximum number of clusters.
    - overlap: Overlap configuration ('random', 'constant', or specific value).
    - max_overlap: Maximum overlap size.
    - matrix_type: Type of matrix ('diagonal' or 'random').

    Returns:
    - block_matrix: The generated block-structured matrix.
    - clusters: List of cluster sizes.
    - overlap_size: Overlap sizes between clusters.
    """
    if cluster == 'random':
        cluster = [random.randint(cluster_size[0], cluster_size[1]) for _ in range(random.randint(min_num_cluster, max_num_cluster))]

    if overlap is None:
        overlap_size = 0
    elif isinstance(overlap, (int, float)):
        overlap_size = overlap
    elif overlap == 'constant':
        overlap_size = random.randint(1, max_overlap) if len(cluster) > 1 else 0
    elif overlap == 'random':
        overlap_size = [random.randint(0, min(cluster) - 1) for _ in range(len(cluster) - 1)]

    n_variables = sum(cluster) - sum(overlap_size) if isinstance(overlap_size, list) else sum(cluster) - overlap_size * (len(cluster) - 1)
    std_devs = [1] * n_variables

    if matrix_type == 'diagonal':
        block_matrix = generate_block_clustered_matrix(n_variables, cluster, overlap_size, std_devs)
    elif matrix_type == 'random':
        block_matrix, cluster = generate_random_block_matrix(min_cluster_size=cluster_size[0], 
                                                             max_cluster_size=cluster_size[1], 
                                                             max_clusters=max_num_cluster)
    
    theoretical_covariance = np.linalg.inv(block_matrix)
    theoretical_precision = block_matrix.copy()
    return theoretical_precision, theoretical_covariance, cluster, overlap_size



def generate_clusters_with_fixed_n_variables(n_variables,
                                             cluster_size_range=[4, 10],
                                             max_overlap=3,
                                             overlap_mode='random',
                                             max_num_clusters=10):
    clusters = []
    overlaps = []

    current_total = 0
    while True:
        if len(clusters) >= max_num_clusters:
            break

        # Scegli una dimensione di cluster casuale
        size = random.randint(cluster_size_range[0], cluster_size_range[1])

        # Calcola overlap con cluster precedente
        if len(clusters) == 0:
            overlap = 0
        else:
            if overlap_mode == 'random':
                overlap = random.randint(0, min(max_overlap, clusters[-1]-1, size-1))
            elif overlap_mode == 'constant':
                overlap = min(max_overlap, clusters[-1]-1, size-1)
            elif isinstance(overlap_mode, int):
                overlap = min(overlap_mode, clusters[-1]-1, size-1)
            else:
                overlap = 0

        # Variabili aggiuntive che questo cluster introduce
        net_new = size - overlap

        # Verifica se aggiungendolo superiamo n_variables
        if current_total + net_new > n_variables:
            break

        clusters.append(size)
        if len(clusters) > 1:
            overlaps.append(overlap)
        current_total += net_new

        if current_total == n_variables:
            break

    # Se serve un ultimo cluster per completare la dimensione
    if current_total < n_variables and len(clusters) < max_num_clusters:
        last_needed = n_variables - current_total
        size = last_needed
        if len(clusters) == 0:
            overlap = 0
        else:
            overlap = 0  # Forza a zero per semplificare
        clusters.append(size)
        if len(clusters) > 1:
            overlaps.append(overlap)
        current_total += (size - overlap)

    if current_total != n_variables:
        raise ValueError("Couldn't generate a consistent clustering for the given n_variables.")

    return clusters, overlaps


def generate_multivariate_samples(n_samples, cov_matrix):
    """
    Generate samples from a multivariate distribution.

    Parameters:
    - n_samples: Number of samples to generate.
    - cov_matrix: Covariance matrix.

    Returns:
    - samples: Generated samples.
    """
    n_vars = cov_matrix.shape[0]
    mean = np.zeros(n_vars)
    samples = np.random.multivariate_normal(mean, cov_matrix, n_samples)
    return samples


def generate_sparse_precision_matrix(N, sparsity=0.95):
    """Generate a sparse precision matrix of size N x N, by a random lower triangular matrix L. and
       sparsity level."""

    #ORIGINAL PAPER CODE

    # Create lower triangular matrix L
    L = np.random.randn(N, N)
    L = np.tril(L)  # Keep only lower triangular part
    
    # Apply sparsity
    mask = np.random.rand(N, N) > sparsity
    L = L * mask
    
    # Ensure diagonal is non-zero for positive definiteness
    L[np.diag_indices(N)] = np.random.uniform(0.5, 2.0, N)
    
    # Create precision matrix: Theta = L @ L.T
    Theta = L @ L.T
    
    # Add small diagonal term for numerical stability
    Theta += np.eye(N) * 0.01
    
    return Theta, np.linalg.inv(Theta)

def main():
    n_variables = 50
    cluster_size_range = [4, 10]
    clusters, overlaps = generate_clusters_with_fixed_n_variables(n_variables,
                                                                cluster_size_range=cluster_size_range,
                                                                max_overlap=3,
                                                                overlap_mode='random')

    print("Clusters:", clusters)
    print("Overlaps:", overlaps)
    print("Total variables:", sum(clusters) - sum(overlaps))


def main():
    n_variables = 10

    # Generazione della matrice con blocchi e overlap
    theoretical_precision, theoretical_covariance, clusters, overlap_size = generate_matrix(
        cluster='random', 
        cluster_size=[4, 10], 
        min_num_cluster=1, 
        max_num_cluster=8, 
        overlap='random', 
        max_overlap=3,
        matrix_type="diagonal"
    )

    # Plot delle due matrici
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    im1 = axs[0].imshow(theoretical_precision, cmap='viridis')
    axs[0].set_title("Theoretical Precision Matrix")
    fig.colorbar(im1, ax=axs[0])

    im2 = axs[1].imshow(theoretical_covariance, cmap='viridis')
    axs[1].set_title("Theoretical Covariance Matrix")
    fig.colorbar(im2, ax=axs[1])

    plt.suptitle(f"Clusters: {clusters} | Overlap size: {overlap_size}", fontsize=10)
    plt.tight_layout()
    plt.show()


def main():
    n_variables = 200
    sparsity = 0.95

    precision_matrix, covariance_matrix = generate_sparse_precision_matrix(n_variables, sparsity)

    # Plot affiancato: Precisione e Covarianza
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Precision matrix
    im1 = axs[0].imshow(precision_matrix, cmap='viridis')
    axs[0].set_title("Sparse Precision Matrix")
    fig.colorbar(im1, ax=axs[0], fraction=0.046, pad=0.04)

    # Covariance matrix
    im2 = axs[1].imshow(covariance_matrix, cmap='viridis')
    axs[1].set_title("Covariance Matrix")
    fig.colorbar(im2, ax=axs[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()

    # Generazione di campioni multivariati
    n_dataset = 30
    T= 800
    samples = FCA.generate_dataset(covariance_matrix, T, n_dataset,  type="Gaussian")

    print("Generated samples shape:", samples.shape)



if __name__ == "__main__":
    main()

