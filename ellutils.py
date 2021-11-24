# coding: utf-8
""" Fonctions utiles pour générer les signaux triphasés à la sortie d'un MZI 
    à trois sortie et estimer les ellipses de polarisation et la phase dans 
    le cas où les signaux triphasés sont mal équilibrés
"""

import numpy as np
import scipy as sp
import scipy.linalg as lin
import matplotlib.pyplot as plt
import warnings


def gen_3_outputs(vamp, vphi, n=100, sig=0.01, theta_start=0, theta_end=2 * np.pi / 3):
    """Genere les trois signaux pour une phase theta qui varie linéairement 
    entre a et b avec les paramètres de deséquilibres en amplitude et phases
    relatives par rapport au cas nominal. Renvoie les signaux observés 
    bruités X, les puissances non bruitées Xwo, et la phase theta"""
    
    theta = np.linspace(theta_start, theta_end, n, endpoint=False)
    #  Paramatres de déséquilibre: vamp et phi
    tri = 2 / 3 * np.pi  # equi angle (rad) for 3 outputs
    p1 = vamp[0] * (1 + np.cos(theta + vphi[0]))
    p2 = vamp[1] * (1 + np.cos(theta + tri + vphi[1]))
    p3 = vamp[2] * (1 + np.cos(theta - tri + vphi[2]))
    # noiseless signal
    Xwo = np.vstack((p1, p2, p3)).T  # n times 3 data matrix
    #  noise
    noise = np.random.normal(0, sig, 3 * n).reshape(n, 3)
    # observed signal (noisy)
    X = Xwo + noise  # n times 3 data matrix
    return X, Xwo, theta


def get_normal_vect(vamp, vphi):
    """Renvoie le vecteur normal au plan de l'ellipse paramétrisée par vamp et
    vphi"""
    
    a1, a2, a3 = vamp
    phi1, phi2, phi3 = vphi
    tri = 2 / 3 * np.pi  # equi angle (rad) for 3 outputs
    betaortho = np.array(
        [
            np.sin(phi2 - phi3 - tri) / a1,
            np.sin(phi3 - phi1 - tri) / a2,
            np.sin(phi1 - phi2 - tri) / a3,
        ]
    ).T
    # Convention to fix the vector sign : sum of the entries is >0
    factnorm = np.sign(np.sum(betaortho)) / np.linalg.norm(betaortho)
    betaortho *= factnorm
    return betaortho.reshape((3, 1))


def get_basis(vamp, vphi):
    """Return Bell, the change of basis orthonormal matrix from the coordinates
    of the 3D signal to the ellipsis plane (u,v) + normal vector (betaortho)
    coordinates"""
    # Compute the vector normal to the ellipse plance
    betaortho = get_normal_vect(vamp, vphi)
    # Projector on the ellipse plane
    Portho = np.eye(3) - betaortho @ betaortho.T
    # Orthonormal basis (u,v) of the ellipse plane
    u = Portho[0, :] / np.linalg.norm(Portho[0, :])
    v = Portho[1, :] - u * np.dot(u, Portho[1, :])
    v /= np.linalg.norm(v)
    # Orthonormal basis (u,v,betaortho)
    # for the ellipe plane (u,v) + normal vector (betaortho)
    Bell = np.vstack((u, v, betaortho.ravel()))
    return Bell


def param2pol(vamp, vphi):
    """Get the polar coefs (a, b, phi, ou, ov, Bell) from
    the paramatric form coefs vamp and vphi
    s.t a >= b and phi in (-pi/2, pi/2)"""

    ## Get the basis where the ellipsis spreads
    Bell = get_basis(vamp, vphi)

    ## Get the ellipsis parameters
    # better to work with complex phasors
    tri = 2 / 3 * np.pi  # equi angle (rad) for 3 outputs
    equiphases = np.array([0, tri, -tri])
    vphasor = np.exp(1j * (vphi + equiphases)) * vamp
    qc = np.sum(vphasor ** 2)
    # fix the initial phasor angle s.t. qc is real
    phase0 = -np.angle(qc) / 2
    vphasor *= np.exp(1j * phase0)
    qc = np.sum(vphasor ** 2)
    ec = np.sum(np.abs(vphasor ** 2))
    qc = np.abs(qc)

    # Get the ellipse axis lengths in the (u,v) plane
    sigmasv = np.sqrt(np.array([(ec + qc) / 2, (ec - qc) / 2]))
    a, b = sigmasv

    # Get the ellipse rotation in the (u,v) plane
    C = np.vstack((np.real(vphasor), np.imag(vphasor))).T
    # Rotation matrix in the (u,v) plane
    R = Bell @ C @ np.diag(1 / sigmasv)[:2, :]
    phi = np.angle(R[0, 0] + 1j * R[1, 0])
    phi = phi - np.pi if phi > np.pi else phi
    phi = phi + np.pi if phi < -np.pi else phi

    # Get the offsets in the (u,v) plane
    ou, ov, o_ortho = Bell @ vamp.reshape(-1, 1)

    return a, b, phi, ou, ov, Bell, phase0


def pol2quad(a, b, phi, ou, ov):
    """Convert the ellipsis geometric/polar parameters (a, b, phi, ou, ov)
    where (ou, ov) is the vector of the ellipis center into quadratic
    form coefficients (A,B,C,D,E,F) s.t.
    Ax**2 + B*x*y + C*y**2 + D*x + E*y + F = 0
    with the convention F=-1
    """
    # Form the rotation and dilation matrix, plus the offset vector
    Rphi = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
    Sab = np.diag([1 / a, 1 / b])
    offvec = np.array([ou, ov]).reshape(-1, 1)

    EllMat = Rphi @ Sab ** 2 @ Rphi.T  # quadratic form matrix
    F = offvec.T @ EllMat @ offvec - 1  # const
    EllMat /= -F[0, 0]  # normalizing convention
    A, B, C = EllMat[0, 0], EllMat[0, 1] * 2, EllMat[1, 1]
    L = -2 * EllMat @ offvec  # linear term
    D, E = L[0, 0], L[1, 0]
    return A, B, C, D, E, -1


def quad2pol(A, B, C, D, E, F):
    """Convert the quadratic form coefficients into ellipsis geometric/polar
    parameters (a, b, phi, ou, ov) where (ou, ov) is the vector of the ellipis
    center with the convention a >= b (u is the major axis) and phi in
    ]-pi/2,pi/2]
    See https://en.wikipedia.org/wiki/Ellipse#General_ellipse"""
    Delta = B ** 2 - 4 * A * C
    if Delta > 0:
        warnings.warn(
            f"(Fitted) Ellipsis matrix is not p.d. : Delta=B^2-4AC={Delta}>0",
            RuntimeWarning,
        )

    a = (
        -np.sqrt(
            2
            * (A * E ** 2 + C * D ** 2 - B * D * E + Delta * F)
            * ((A + C) + np.sqrt((A - C) ** 2 + B ** 2))
        )
        / Delta
    )
    b = (
        -np.sqrt(
            2
            * (A * E ** 2 + C * D ** 2 - B * D * E + Delta * F)
            * ((A + C) - np.sqrt((A - C) ** 2 + B ** 2))
        )
        / Delta
    )

    ou = (2 * C * D - B * E) / Delta
    ov = (2 * A * E - B * D) / Delta

    if np.isclose(B, 0):
        phi = 0 if (A < C) else np.pi / 2
    else:
        phi = np.arctan((C - A - np.sqrt((A - C) ** 2 + B ** 2)) / B)

    # swap the ellipsis axes to ensure that vector u is the major axis
    if a < b:
        a, b = b, a
        phi = phi - np.pi / 2 if phi > 0 else phi + np.pi / 2

    return a, b, phi, ou, ov


def pol2balance(a, b, phi, Bell, phaseref=0):
    """Convert the ellipsis geometric/polar parameters (a, b, phi)
    into the angle and modulus of the unbalanced phasor/modulators
    for our measured signal.
    Bell is the change of basis orthogonal matrix from the
    coordinate of the 3D signal to the (ellipsis plane + normal vector)
    coordinate
    """
    #  Rotate and dilate the analytic signal (cos(theta), sin(theta))
    Rphi = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
    Sab = np.diag([a, b])
    # Complex (stationnary) phasor+modulator
    Cphasor = Bell[0:2, :].T @ Rphi @ Sab @ ([[1], [1j]])
    modpower = np.abs(Cphasor).ravel()  # 3D vector (a1hat, a2hat, a3hat)
    phases = np.angle(Cphasor).ravel()  # 3D vector (phi1hat, phi2hat, phi3hat)
    # we can fix first relative phase arbitrarily to phaseref
    phase0 = phases[0]
    phases -= phase0 - phaseref
    return modpower, phases, phase0 - phaseref

def validation_score(A, B, C, D, E, F, offset, Bell):
    """Compare the estimated offsets with the estimated amplitudes
    of the power pi = ai_hat cos(theta+phii_hat) +offseti_hat,
    for i=1,2,3. In our physical model, this quantities are the same.
    This returns ||a_hat - offset_hat||_2 that is used as a
    goodness-of-fit measure (equal to zero for a perfect fit)
    """
    if (B ** 2 - 4 * A * C) > 0:
        # print("not an ellipsis")
        return +np.inf

    a, b, phi, offu, offv = quad2pol(A, B, C, D, E, F)
    modpower, _, _ = pol2balance(a, b, phi, Bell, phaseref=0)
    offvect = np.array([[offu, offv, offset]]).ravel()
    modpoweroff = (Bell.T @ offvect).ravel()
    # print(f"vel socre func : resnorm =  {lin.norm(modpower-modpoweroff)}")
    return lin.norm(modpower - modpoweroff)


def pol2theta(a, b, phi, offu, offv, Bell, phase0, X):  # add offu, offv, phase0
    """
    Correct the unbalanced three phase signals to get the angle theta
    """
    #  Rotate and dilate the analytic signal (cos(theta), sin(theta))
    Rphi = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
    Sabinv = np.diag([1 / a, 1 / b])
    Xplan = (Bell @ X.T)[0:2, :] - np.array([offu, offv]).reshape(
        2, 1
    )  # 2 times n matrix
    zanalytic = Sabinv @ Rphi.T @ Xplan  # (cos(theta), sin(theta))
    theta = np.angle(zanalytic[0, :] - 1j * zanalytic[1, :])
    #theta += phase0 - phi1  # the arbitrary origin for the first phasor
    theta += phase0  # the arbitrary origin for the first phasor
    theta = np.unwrap(theta)
    return theta


def subsangle(x, y):
    """Substract two arrays of angle, i.e. x-y.
    This returns an array of the same dimension with angles in [-pi,pi["""
    r = 2 * np.pi
    a = np.asarray((x - y) % r)
    b = np.asarray((y - x) % r)
    res = b
    inda = a < b
    res[inda] = -a[inda]
    return res