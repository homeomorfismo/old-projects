"""
Testing error estimator implementation.
Laplacian eigenvalue problem on a L-shaped domain.
    - Δu = λu in Ω
Dual weighted residual error estimator.
"""
import math
from functools import partial
import ngsolve as ng
import numpy as np
from pyeigfeast import NGvecs, SpectralProjNG
from netgen.geom2d import SplineGeometry
from ngs_ee.estimator import Estimator
from ngs_ee.adap_strat import AdaptivityStrategy

# Parameters for the spectral projection
NSPAN = 4
RADIUS = 0.01
CENTER = 8*math.pi**2
NPTS = 6
CHECKS = True
WITHIN = None
RHOINV = 0.0
QUADRULE = 'circ_trapez_shift'
INVERSE = 'umfpack'
VERBOSE = True
N_RESTART = 2
N_ITERS = 100
# Other parameters
MAX_DOF = 100000
ORDER = 3


def make_geometry() -> SplineGeometry:
    """
    Make a L-shaped domain with a corner singularity at (0,0).
    """
    geometry = SplineGeometry()
    points = [(0, 0), (1, 0), (1, 1), (-1, 1), (-1, -1), (0, -1)]
    p_nums = [geometry.AppendPoint(*p) for p in points]
    # (start, end, bc, left, right)
    lines = [(0, 1, 1, 1, 0), (1, 2, 1, 1, 0), (2, 3, 1, 1, 0),
             (3, 4, 1, 1, 0), (4, 5, 1, 1, 0), (5, 0, 1, 1, 0)]
    for p1, p2, bc, left, right in lines:
        geometry.Append(
                ["line", p_nums[p1], p_nums[p2]],
                bc=bc,
                leftdomain=left,
                rightdomain=right)
    return geometry


def make_mesh(
        geometry: SplineGeometry,
        maxh: float) -> ng.Mesh:
    """
    Make a mesh from the geometry.
    """
    mesh = ng.Mesh(geometry.GenerateMesh(maxh=maxh))
    return mesh


def make_fespace(
        mesh: ng.Mesh,
        order: int = 1) -> [ng.FESpace, ng.GridFunction]:
    """
    Make a finite element space.
    """
    fes = ng.H1(
        mesh,
        order=order,
        dirichlet=".*",
        complex=True,
        autoupdate=True)
    solution = ng.GridFunction(fes, "solution", autoupdate=True)
    return fes, solution


def make_matrices(
        fes: ng.FESpace) -> [ng.BilinearForm, ng.BilinearForm]:
    """
    Make the matrices for the Laplacian eigenvalue problem.
    """
    u = fes.TrialFunction()
    v = fes.TestFunction()
    a = ng.BilinearForm(fes, symmetric=True)
    a += ng.grad(u) * ng.grad(v) * ng.dx
    b = ng.BilinearForm(fes, symmetric=True)
    b += u * v * ng.dx
    assemble(a, b)
    return a, b


def assemble(*args) -> None:
    """
    Update the matrices for the Laplacian eigenvalue problem.
    """
    with ng.TaskManager():
        try:
            for arg in args:
                arg.Assemble()
        except Exception as e:
            print(e)
            print("\n\tTrying with larger heap size...")
            ng.SetHeapSize(int(1e9))
            for arg in args:
                arg.Assemble()


def dual_weighted_residual(
        fes: ng.FESpace,
        input_value: tuple) -> np.ndarray:
    """
    Dual weighted residual error estimator.
    Use Partial Application to pass the finite element space.
    """
    def laplacian(u):
        grad_u = ng.grad(u)
        return grad_u[0].Diff(ng.x) + grad_u[1].Diff(ng.y)

    input_value = list(input_value)
    assert len(input_value) == 3, f"Expected 3 values, got {len(input_value)}"

    right_solution = input_value[0]
    left_solution = input_value[1]
    eigenvalue = input_value[2]

    n = ng.specialcf.normal(fes.mesh.dim)
    h = ng.specialcf.mesh_size

    # Residuals and jumps
    right_residual = h * (laplacian(right_solution) +
                          eigenvalue * right_solution)
    right_jump = ng.sqrt(0.5 * h) * \
        (ng.grad(right_solution) - ng.grad(right_solution).Other()) * n
    left_residual = h * (laplacian(left_solution) + eigenvalue * left_solution)
    left_jump = ng.sqrt(0.5 * h) * \
        (ng.grad(left_solution) - ng.grad(left_solution).Other()) * n
    # Weights
    right_weight = ng.grad(left_solution)
    left_weight = ng.grad(right_solution)
    # Dual weighted residual
    rho_r = ng.Integrate(
        ng.InnerProduct(right_residual, right_residual) * ng.dx +
        ng.InnerProduct(right_jump, right_jump) *
        ng.dx(element_boundary=True),
        fes.mesh,
        element_wise=True)
    rho_l = ng.Integrate(
        ng.InnerProduct(left_residual, left_residual) * ng.dx +
        ng.InnerProduct(left_jump, left_jump) *
        ng.dx(element_boundary=True),
        fes.mesh,
        element_wise=True)
    omega_r = ng.Integrate(
        ng.InnerProduct(right_weight, right_weight) * ng.dx,
        fes.mesh,
        element_wise=True)
    omega_l = ng.Integrate(
        ng.InnerProduct(left_weight, left_weight) * ng.dx,
        fes.mesh,
        element_wise=True)

    eta = np.sqrt(rho_r.real.NumPy() * omega_r.real.NumPy())
    eta += np.sqrt(rho_l.real.NumPy() * omega_l.real.NumPy())
    return eta


def test_adaptivity() -> None:
    """
    Test the error estimator.
    """
    geometry = make_geometry()
    mesh = make_mesh(geometry, maxh=0.25)
    fes, _ = make_fespace(mesh, order=ORDER)
    a, b = make_matrices(fes)
    dwr = partial(dual_weighted_residual, fes)
    estimator = Estimator(
            dwr, AdaptivityStrategy.GREEDY,
            'Dual Weighted Residual with Greedy Strategy')
    while True:
        # Assemble matrices
        assemble(a, b)
        # Set eigenspace
        right_espan = NGvecs(fes, NSPAN)
        left_espan = NGvecs(fes, NSPAN)
        right_espan.setrandom(seed=1)
        left_espan.setrandom(seed=1)

        proj = SpectralProjNG(
            fes,
            a.mat,
            b.mat,
            radius=RADIUS,
            center=CENTER,
            npts=NPTS,
            within=WITHIN,
            rhoinv=RHOINV,
            checks=CHECKS,
            quadrule=QUADRULE,
            verbose=VERBOSE,
            inverse=INVERSE)

        # Compute eigenvalues
        eigenvalues, right_espan, history, left_espan = proj.feast(
            right_espan,
            Yl=left_espan,
            hermitian=False,
            nrestarts=N_RESTART,
            niterations=N_ITERS)

        if not history[-1]:
            raise ValueError("No convergence in FEAST.")

        if fes.ndof > MAX_DOF:
            break

        estimator.mark(
            mesh,
            *((right_espan[i], left_espan[i], eigenvalues[i])
              for i in range(len(eigenvalues))))
        mesh.Refine()
    if __name__ == "__main__":
        right_espan.draw()
    print("Test passed.")


if __name__ == "__main__":
    test_adaptivity()
