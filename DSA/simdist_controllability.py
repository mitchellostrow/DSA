from typing import Literal
import numpy as np
from scipy.linalg import orthogonal_procrustes

try:
    from .simdist import SimilarityTransformDist
except ImportError:
    from simdist import SimilarityTransformDist 

class ControllabilitySimilarityTransformDist:
    """
    Procrustes analysis over vector fields / LTI systems.
    Only Euclidean scoring is implemented in this closed-form version.
    """
    def __init__(
        self,
        *,
        score_method: Literal["euclidean", "angular"] = "euclidean",
        compare: Literal['joint','control','state'] = 'joint',
        joint_optim: bool = False,
        return_distance_components=True
    ):
        f"""
        Parameters
        ----------
        score_method : {"euclidean", "angular"}
            Distance method to use. Euclidean uses Frobenius norm, angular uses principal angles.
        compare: {'joint','control','state'}
            what type of comparison to do on the A and B matrices
        align_inputs : bool
            If True, do two-sided Procrustes on controllability matrices (solve for C and C_u).
        """
        self.score_method = score_method
        self.compare = compare
        self.joint_optim = joint_optim
        self.return_distance_components=return_distance_components


    def fit_score(self, A, B, A_control, B_control):
        
        C, C_u, sims_joint_euc, sims_joint_ang = self.compare_systems_procrustes(
            A1=A, B1=A_control, A2=B, B2=B_control, align_inputs=self.joint_optim
        )
 

        score_method = self.score_method

        if self.compare == 'joint':
            if self.return_distance_components:
                if self.score_method == 'euclidean':
                    # sims_control_joint = np.linalg.norm(C @ A_control @ C_u - B_control, "fro") ** 2
                    # sims_state_joint = np.linalg.norm(C @ A @ C.T - B, "fro") ** 2
                    sims_control_joint = np.linalg.norm(C @ A_control @ C_u - B_control, "fro") 
                    sims_state_joint = np.linalg.norm(C @ A @ C.T - B, "fro") 
                    return sims_joint_euc, sims_state_joint, sims_control_joint
                elif self.score_method == 'angular':
                    sims_control_joint = np.trace((C @ A_control @ C_u).T @ B_control) / (np.linalg.norm(C @ A_control @ C_u, 'fro') * np.linalg.norm(B_control, 'fro'))
                    sims_state_joint = np.trace((C @ A @ C.T).T @ B) / (np.linalg.norm(C @ A @ C.T, 'fro') * np.linalg.norm(B, 'fro'))

                    sims_control_joint = np.clip(sims_control_joint, -1, 1)
                    sims_state_joint = np.clip(sims_state_joint, -1, 1)
                    sims_control_joint =np.arccos(sims_control_joint)
                    sims_state_joint =np.arccos(sims_state_joint)
                    sims_control_joint = np.clip(sims_control_joint, 0, np.pi)
                    sims_state_joint = np.clip(sims_state_joint, 0, np.pi)

                    return sims_joint_ang, sims_state_joint, sims_control_joint
            else:
                if self.score_method == 'euclidean':
                    return sims_joint_euc
                elif self.score_method == 'angular':
                    return sims_joint_ang
                else:
                    raise ValueError('Choose between Euclidean or angular distance')

        elif self.compare:
            return self.compare_A(A, B, score_method=score_method)

        else:
            return self.compare_B(A_control, B_control, score_method=score_method)


    def get_controllability_matrix(self, A, B):
        """
        Computes the controllability matrix K = [B, AB, A^2B, ..., A^(n-1)B].

        Args:
            A (np.ndarray): The state matrix (n x n).
            B (np.ndarray): The input matrix (n x m).

        Returns:
            np.ndarray: The controllability matrix (n x n*m).
        """
        n = A.shape[0]
        K = B.copy()
        current1_term = B.copy()  # Start with A^0 * B = B
        current2_term = B.copy()  # Start with A^0 * B = B
        
        for i in range(1, n):
            # current_term = np.linalg.matrix_power(A, i) @ B  # Use stable matrix power function
            current1_term = A @ current1_term
            current2_term = A.T @ current2_term
            
            # Check for numerical instability
            # term_norm = np.linalg.norm(current_term)
            # if term_norm < 1e-12 or term_norm > 1e12:
                # break
                
            # Check for linear dependence (rank deficiency)
            K_test = np.hstack((K, current1_term, current2_term))
            # if np.linalg.matrix_rank(K_test) <= np.linalg.matrix_rank(K):
                # break
                
            K = K_test
        return K

    def compare_systems_procrustes(self, A1, B1, A2, B2, *,align_inputs=False):
        """
        Compares two LTI systems by finding the optimal orthogonal transformation
        that aligns their controllability matrices.

        This implements the fast, non-iterative solution to the Orthogonal
        Procrustes problem.

        Args:
            A1, B1 (np.ndarray): Matrices for the first system.
            A2, B2 (np.ndarray): Matrices for the second system.
            align_inputs (bool): If True, do two-sided Procrustes (not used in updated version).
            n (int): Number of terms in controllability matrix.

        Returns
        -------
        C : (n×n) orthogonal state transform
        C_u : (p×p) orthogonal input/right transform (identity in updated version)
        err : Frobenius residual
        cos_sim : cosine similarity between K1 and aligned K2
        """
        # Build controllability matrices: K \in R^{n x p}
        K1 = self.get_controllability_matrix(A1, B1)
        K2 = self.get_controllability_matrix(A2, B2)

        if not align_inputs:
            # One-sided: C = argmin ||K1 - C K2||_F
            M = K2 @ K1.T
            U, _, Vh = np.linalg.svd(M, full_matrices=False)
            C = U @ Vh
            K2_aligned = C @ K2
            err = np.linalg.norm(K1 - K2_aligned, "fro")
            cos_sim = (np.vdot(K1, K2_aligned).real /
                       (np.linalg.norm(K1, "fro") * np.linalg.norm(K2, "fro")))
            cos_sim = np.clip(cos_sim, -1, 1)
            cos_sim = np.arccos(cos_sim)
            cos_sim = np.clip(cos_sim, 0, np.pi)
            return C, np.eye(B2.shape[-1]), err, cos_sim

        # Two-sided: C, C_u = argmin ||K1 - C K2 C_u||_F
        U1, S1, V1t = np.linalg.svd(K1, full_matrices=False)
        U2, S2, V2t = np.linalg.svd(K2, full_matrices=False)

        C   = U1 @ U2.T
        C_u = V2t.T @ V1t  # = V2 @ V1^T
        
        K2_aligned = C @ K2 @ C_u
        err = np.linalg.norm(K1 - K2_aligned, "fro")
        cos_sim = (np.vdot(K1, K2_aligned).real /
                   (np.linalg.norm(K1, "fro") * np.linalg.norm(K2, "fro")))
        cos_sim = np.clip(cos_sim, -1, 1)
        cos_sim = np.arccos(cos_sim)
        cos_sim = np.clip(cos_sim, 0, np.pi)
        
        return C, C_u, err, cos_sim

    @staticmethod
    def compare_A(A1, A2, score_method='euclidean'):
        simdist = SimilarityTransformDist(iters=1000, score_method=score_method, lr=1e-3, verbose=True)
        return simdist.fit_score(A1, A2, score_method=score_method)

    @staticmethod
    def compare_B(B1, B2, score_method='euclidean'):
        if score_method == 'euclidean':
            R, _ = orthogonal_procrustes(B2.T, B1.T)
            return np.linalg.norm(B1 - R.T @ B2, "fro") 
            # return np.linalg.norm(B1 - R.T @ B2, "fro") ** 2
        elif score_method == 'angular':
            return np.trace(B1.T @ (R.T @ B2)) / (np.linalg.norm(B1, 'fro') * np.linalg.norm(R.T @ B2, 'fro'))
        else:
            raise ValueError('Choose between Euclidean or angular distance')

