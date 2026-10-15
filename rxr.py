"""
Educational PN (spherical-harmonics) solver for a 1-D radial cylindrical
reactor – steady-state, one-group, isotropic scattering.

This script is intentionally transparent for educational purposes, allowing
students to:
  - See the PN expansion and moment equations.
  - Understand the impact of cylindrical geometry terms.
  - Experiment with the PN order (N) and observe solution changes.
  - Inspect residuals and convergence diagnostics.

Author: C.J. Miko
"""
import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

# Suppress specific runtime warnings for cleaner output.
warnings.filterwarnings("ignore", category=RuntimeWarning)


def define_problem():
    """
    Defines the geometry, mesh, and material properties for the reactor problem.

    Returns:
        tuple: A tuple containing problem parameters:
               (R, NUM_CELLS, N_ORDER, SIGMA_T, SIGMA_S, NU_SIGMA_F)
    """
    R = 50.0            # Cylinder radius [cm]
    NUM_CELLS = 1000    # Number of radial cells
    N_ORDER = 1         # PN order (currently supports P1)
    SIGMA_T = 1.0       # Total cross-section [1/cm]
    SIGMA_S = 0.9       # Isotropic scatter cross-section [1/cm]
    NU_SIGMA_F = 0.10   # Fission production cross-section [1/cm]
    return R, NUM_CELLS, N_ORDER, SIGMA_T, SIGMA_S, NU_SIGMA_F


def build_p1_system(params):
    """
    Builds the P1 (diffusion) theory matrix system for a cylindrical reactor.

    The method discretizes the diffusion equation using a cell-centered finite
    volume approach on a radial mesh.
    -∇·D∇φ + Σa φ = (1/k) νΣf φ

    Args:
        params (dict): A dictionary of problem parameters.

    Returns:
        tuple: A tuple containing the sparse matrices (A, F), where A is the
               destruction/transport operator and F is the production operator.
    """
    # Unpack parameters
    R, NUM_CELLS, SIGMA_T, SIGMA_S, NU_SIGMA_F = (
        params['R'], params['NUM_CELLS'], params['SIGMA_T'],
        params['SIGMA_S'], params['NU_SIGMA_F']
    )
    r_edges = params['r_edges']
    dr = params['dr']
    sigma_a = SIGMA_T - SIGMA_S
    # Transport-corrected diffusion coefficient
    D = 1.0 / (3.0 * SIGMA_T)

    print(f"Diffusion coefficient D = {D:.5f} cm")

    A = lil_matrix((NUM_CELLS, NUM_CELLS))
    F = lil_matrix((NUM_CELLS, NUM_CELLS))

    for i in range(NUM_CELLS):
        vol = np.pi * (r_edges[i+1]**2 - r_edges[i]**2)
        A[i, i] += sigma_a * vol
        F[i, i] += NU_SIGMA_F * vol

        # Left face coupling (i-1)
        if i > 0:
            r_face = r_edges[i]
            area = 2 * np.pi * r_face
            coup = area * D / dr
            A[i, i] += coup
            A[i, i-1] -= coup

        # Right face coupling (i+1)
        if i < NUM_CELLS - 1:
            r_face = r_edges[i+1]
            area = 2 * np.pi * r_face
            coup = area * D / dr
            A[i, i] += coup
            A[i, i+1] -= coup
        else:  # Vacuum boundary condition at the outer edge
            d = 0.7104 / SIGMA_T  # Extrapolated length
            r_face = R
            area = 2 * np.pi * r_face
            # Effective coupling based on extrapolated zero-flux condition
            coup = area * D / (dr / 2 + d)
            A[i, i] += coup

    return A.tocsr(), F.tocsr()


def solve_eigenvalue_pi(A, F, num_cells, max_iter=200, tol=1e-7):
    """
    Solves the generalized eigenvalue problem A φ = (1/k) F φ using power iteration.

    Args:
        A (csr_matrix): The destruction/transport operator.
        F (csr_matrix): The production operator.
        num_cells (int): The number of cells in the mesh.
        max_iter (int): Maximum number of iterations.
        tol (float): Convergence tolerance for k-eff.

    Returns:
        tuple: A tuple containing the final k_eff and the corresponding
               eigenvector (scalar flux phi).
    """
    k_eff = 1.0
    k_prev = 1.0
    phi = np.ones(num_cells)
    phi /= np.linalg.norm(phi)

    for it in range(max_iter):
        rhs = F @ phi
        psi = spsolve(A, rhs)

        # Rayleigh quotient for a better k-eff estimate
        k_eff = (psi @ (F @ psi)) / (psi @ (A @ psi))
        
        # Check for convergence
        if abs(k_eff - k_prev) < tol:
            print(f"Power iteration converged after {it+1} iterations.")
            break
        
        phi = psi / np.linalg.norm(psi)
        k_prev = k_eff

        if (it + 1) % 20 == 0:
            residual = np.linalg.norm(A @ phi - (1/k_eff) * F @ phi)
            print(f"Iter {it+1:3d}:  k-eff ≈ {k_eff:.6f}   residual = {residual:.2e}")

    return k_eff, phi


def plot_results(params, k_eff, phi):
    """
    Generates and displays plots for flux, current, and residuals.

    Args:
        params (dict): A dictionary of problem parameters.
        k_eff (float): The calculated effective multiplication factor.
        phi (np.ndarray): The scalar flux distribution.
    """
    r_centers = params['r_centers']
    n_order = params['N_ORDER']
    D = 1.0 / (3.0 * params['SIGMA_T'])
    A = params['A']
    F = params['F']

    scalar_flux = phi / phi.max()
    current = -D * np.gradient(scalar_flux, r_centers)
    res = A @ phi - (1/k_eff) * F @ phi

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle(f"P{n_order} Cylindrical Reactor Results (k-eff = {k_eff:.6f})")

    # Scalar Flux (l=0)
    axes[0, 0].plot(r_centers, scalar_flux, 'b-', lw=2)
    axes[0, 0].set(xlabel="Radius r [cm]", ylabel="Normalized scalar flux φ₀",
                   title="Scalar Flux (PN moment l=0)", xlim=(0, None))
    axes[0, 0].grid(True)

    # Radial Current (l=1)
    axes[0, 1].plot(r_centers, current, 'r-', lw=2)
    axes[0, 1].set(xlabel="Radius r [cm]", ylabel="Net current J_r ≈ φ₁",
                   title="Radial Current (PN moment l=1)", xlim=(0, None))
    axes[0, 1].grid(True)

    # Residual Map
    axes[1, 0].semilogy(r_centers, np.abs(res), 'k-')
    axes[1, 0].set(xlabel="Radius r [cm]", ylabel="|Residual|",
                   title="Local Residuals", xlim=(0, None))
    axes[1, 0].grid(True)
    
    # Explanatory Text
    pn_text = (
        f"PN order used: {n_order}\n"
        f"(This run is P1 = diffusion)\n\n"
        "To go to P3/P5, you would:\n"
        "1. Expand ψ(r,μ) ≈ Σ φ_l(r) P_l(μ).\n"
        "2. Insert into the transport equation.\n"
        "3. Project onto each Legendre polynomial P_l.\n"
        "4. Obtain a coupled system of ODEs for φ_l.\n"
        "5. Discretize and solve the larger block matrix."
    )
    axes[1, 1].text(0.1, 0.5, pn_text, transform=axes[1, 1].transAxes,
                    fontsize=10, verticalalignment='center',
                    fontfamily='monospace')
    axes[1, 1].axis('off')
    axes[1, 1].set_title("How to Raise the PN Order")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def main():
    """Main execution function."""
    R, NUM_CELLS, N_ORDER, SIGMA_T, SIGMA_S, NU_SIGMA_F = define_problem()

    # Derived parameters
    sigma_a = SIGMA_T - SIGMA_S
    dr = R / NUM_CELLS
    r_edges = np.linspace(0.0, R, NUM_CELLS + 1)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])

    params = {
        'R': R, 'NUM_CELLS': NUM_CELLS, 'N_ORDER': N_ORDER,
        'SIGMA_T': SIGMA_T, 'SIGMA_S': SIGMA_S, 'NU_SIGMA_F': NU_SIGMA_F,
        'dr': dr, 'r_edges': r_edges, 'r_centers': r_centers
    }

    print(f"Geometry: infinite cylinder, R = {R} cm")
    print(f"Mesh: {NUM_CELLS} cells, dr = {dr:.4f} cm")
    print(f"PN order N = {N_ORDER}")
    print(f"XS: Σt={SIGMA_T}, Σs={SIGMA_S}, Σa={sigma_a:.2f}, νΣf={NU_SIGMA_F}")
    print("-" * 60)

    A, F = build_p1_system(params)
    params['A'], params['F'] = A, F  # Store matrices for plotting

    k_eff, phi = solve_eigenvalue_pi(A, F, NUM_CELLS)

    print("-" * 60)
    print(f"Converged k-eff (P1 / diffusion) = {k_eff:.6f}")

    # Analytic estimate for infinite cylinder (diffusion)
    D = 1.0 / (3.0 * SIGMA_T)
    R_ex = R + 0.7104 / SIGMA_T  # Extrapolated radius
    Bg2 = (2.4048255577 / R_ex)**2  # Geometric buckling
    k_analytic = NU_SIGMA_F / (sigma_a + D * Bg2)

    print(f"Analytic diffusion k-eff         = {k_analytic:.6f}")
    rel_diff = abs(k_eff - k_analytic) / k_analytic * 100
    print(f"Relative difference              = {rel_diff:.3f} %")
    print("-" * 60)
    
    plot_results(params, k_eff, phi)


if __name__ == "__main__":
    main()

