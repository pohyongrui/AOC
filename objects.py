# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 13:44:10 2023

@author: yongruipoh
"""

import os
import sys
import platform
import numpy as np
import scipy as sp
import warnings
import dill
import pathos.multiprocessing as mp
import seaborn as sns
import pandas as pd
import qutip
import matplotlib.pyplot as plt
import matplotlib.cm as cm
### Choose alternative font on supercomputer ###
if platform.system() == "Linux":
    font = "Nimbus Sans"
else:
    font = "Arial"
plt.rcParams.update({
    "text.usetex": False,
    "font.family": font,
    "mathtext.fontset": "custom",
    "mathtext.rm": font,
    "mathtext.it": f"{font}:italic",
    "axes.formatter.useoffset": False
})

def wavenumber_to_Hz(quantity):
    return quantity * 29979245800
    
def Hz_to_wavenumber(quantity):
    return quantity * 3.335641E-11

def lebedev_rule(order):
    """
    A function that works exactly like scipy.integrate.lebedev_rule.
    
    Context: scipy.integrate.lebedev_rule is only available in the latest SciPy version. Hence, for compatibility
                with earlier versions of SciPy, the library has been extracted and stored as a dictionary in 
                lebedev_rule.pkl.
    """
    
    with open("lebedev_rule.pkl", "rb") as file:
        dictionary = dill.load(file)
    
    try:
        return dictionary[order]
    except KeyError:
        raise NotImplementedError(f"Order n={order} not available. Available orders are [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 35, 41, 47, 53, 59, 65, 71, 77, 83, 89, 95, 101, 107, 113, 119, 125, 131].")
    
def get_rot_mat_b_to_a(a, b):
    """
    Returns the rotation matrix that aligns vector b to vector a.
    Uses scipy.spatial.transform.Rotation.align_vectors.
    
    Attributes:
        a, b: Vectors
        
    Returns:
        rot_matrix: Rotation matrix as a NumPy array
    """
    
    a, b = np.array([a, ]), np.array([b, ])     # scipy.spatial.transform.Rotation.align_vectors only accepts arrays of size (N, 3) as input
    
    # Ignore warning associated with rotating two vectors rather than two matrices
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rot_matrix, _ = sp.spatial.transform.Rotation.align_vectors(a, b)
        rot_matrix = rot_matrix.as_matrix()
    
    # Check result
    new_a = rot_matrix @ b[0]
    if np.allclose(a[0], new_a):
        return rot_matrix
    else:
        raise ValueError(f"Rotation matrix could not be generated accurately. \nOld vector a: {a[0]} \nNew vector a: {new_a}")

def get_rot_mat_about_a(a, N):
    """
    Returns the rotation matrix that rotates about vector a by an angle of 2 * pi / N.
    Uses scipy.spatial.transform.Rotation.from_rotvec.
    
    Attributes:
        a: Vector
        N: Total number of rotations
        
    Returns:
        rot_matrix: Rotation matrix as a NumPy array
    """
    
    angle = 2 * np.pi / N
    a = a / np.linalg.norm(a)   # Normalise a
    rot_matrix = sp.spatial.transform.Rotation.from_rotvec(angle * a)
    rot_matrix = rot_matrix.as_matrix()
    
    return rot_matrix

class HiddenPrints:
    """
    Hides all prints.
    Usage:
        with HiddenPrints():
            print("This will not be printed")

        print("This will be printed as before")
    
    Taken from https://stackoverflow.com/questions/8391411/how-to-block-calls-to-print
    """
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

class Dynamics:
    """
    Simulates the excited-state dynamics using a Lindblad master equation model.
    The Liouvillian is propagated using QuTiP.
    *** NOTE: Only the excited-state decay is modelled here ***
    
    Attributes:
        E: Average excited-state energy (cm^{-1})
        Delta: Excited-state energy splitting due to in-state SOC (cm^{-1})
        g_L: Excited-state orbital g-factor
        B: Applied magnetic field strength (T)
        C: Excited-to-ground electronic coupling (cm^{-1})
        D: Excited-to-ground SOC (cm^{-1})
        t_list: NumPy array of timepoints to be passed into QuTiP (s)
        
        **kwargs
        b: Numpy array of floats describing vector [b_x, b_y, b_z] pointing along the applied magnetic field.
            Note that this vector will be normalised.
            Defaults to [0.,0.,1.].
        e: Numpy array of floats describing vector [e_x, e_y, e_z] corresponding to the light polarisation.
            Note that this vector should be perpendicular to the direction of light propagation.
            Note that this vector will be normalised.
            Defaults to [0.,1.,0.].            
        progress_bar: Boolean describing whether to print the progress of each simulation on QuTiP. Defaults to True.
        G: Float characterising the additional electronic coupling between the two OAM states (cm^{-1}).
            Defaults to zero.
        rho_0: QuTiP object describing the initial state.
                Defaults to None, which calls the set of superposition states generated by the optical light pulse.
        tau_exc: Excited state lifetime (s). Defaults to 190E-9.
        no_trvs_Zee: Boolean describing whether to switch off transverse Zeeman effects (for testing purposes).
                Defaults to False.
    """
    
    ### Define physical constants ###
    mu_B = wavenumber_to_Hz(0.466864)  # in units of Hz T^{-1}
    g_e = 2.002319
    
    ### Define the Hilbert space ###
    zero_a, zero_b = qutip.basis(6,0), qutip.basis(6,1)
    plus_b, minus_a = qutip.basis(6,2), qutip.basis(6,3)
    plus_a, minus_b = qutip.basis(6,4), qutip.basis(6,5)
    
    @classmethod
    def get_g_mag(cls, rho, b):
        """
        Computes the ground-state magnetisation for a given density matrix.
        
        Attributes:
            rho: Density matrix.
            b: Unit vector pointing along the applied magnetic field.
            
        Returns:
            mag: Magnetisation, computed as Tr(rho * s)
        """
        
        ### Extract elements of b ###
        b_x, b_y, b_z = b
        
        # ### Define the magnetisation operator ###
        # s_x = 0.5 * (cls.zero_a * cls.zero_b.dag() + cls.zero_b * cls.zero_a.dag())
        # s_y = 0.5 * (-1j * cls.zero_a * cls.zero_b.dag() + 1j * cls.zero_b * cls.zero_a.dag())
        # s_z = 0.5 * (cls.zero_a * cls.zero_a.dag() - cls.zero_b * cls.zero_b.dag())
        # s = s_x * b_x + s_y * b_y + s_z * b_z
        
        # ### Compute magnetisation ###
        # # g_mag = (rho * s).tr().real     # rho and s are both Hermitian.
        # g_mag = qutip.expect(s, rho)
        
        ### Extract the density operator within the reduced Hilbert space ###
        rho = qutip.Qobj([[rho[0,0], rho[0,1]],
                            [rho[1,0], rho[1,1]]])
        
        ### Define the magnetisation operator ###
        s = 0.5 * (qutip.sigmax() * b_x + qutip.sigmay() * b_y + qutip.sigmaz() * b_z)
        
        ### Compute magnetisation ###
        # g_mag = (rho * s).tr().real     # rho and s are both Hermitian.
        g_mag = qutip.expect(s, rho)
        
        return g_mag
        
    def __init__(self,
                 E: float,
                 Delta: float,
                 g_L: float,
                 B: float,
                 C: float,
                 D: float,
                 t_list: np.ndarray,
                 **kwargs):
        
        self.E = E
        self.Delta = Delta
        self.g_L = g_L
        self.B = B
        self.C = C
        self.D = D
        self.t_list = t_list
        try:
            self.b = kwargs["b"]
        except KeyError:
            self.b = np.array([0.,0.,1.])
        try:
            self.e = kwargs["e"]
        except KeyError:
            self.e = np.array([0.,1.,0.])
        try:
            self.progress_bar = kwargs["progress_bar"]
        except KeyError:
            self.progress_bar = True
        try:
            self.G = kwargs["G"]
        except KeyError:
            self.G = 0.
        try:
            self.rho_0 = kwargs["rho_0"]
        except KeyError:
            self.rho_0 = None
        try:
            self.tau_exc = kwargs["tau_exc"]
        except KeyError:
            self.tau_exc = 190E-9
        try:
            self.no_trvs_Zee = kwargs["no_trvs_Zee"]
        except KeyError:
            self.no_trvs_Zee = False
            
        # Check that all inputs are valid
        if self.t_list.ndim != 1:
            raise TypeError(f"t_list should be a one-dimensional NumPy array.")
        if not isinstance(self.b, np.ndarray) or not all(isinstance(b_j, float) for b_j in self.b):
            raise TypeError(f"b should be entered as a NumPy array of floats.")
        self.b = self.b / np.linalg.norm(self.b)    # Normalise b
        if not isinstance(self.e, np.ndarray) or not all(isinstance(e_j, float) for e_j in self.e):
            raise TypeError(f"e should be entered as a NumPy array of floats.")
        self.e = self.e / np.linalg.norm(self.e)    # Normalise e
        if not isinstance(self.progress_bar, bool):
            raise TypeError(f"progress_bar should be entered as a boolean.")
        if not isinstance(self.G, float):
            raise TypeError(f"G should be entered as a float.")
        if not isinstance(self.rho_0, qutip.qobj.Qobj) and not self.rho_0 is None:
            raise TypeError(f"rho_0 should be entered as a QuTiP object.")
        if not isinstance(self.tau_exc, float):
            raise TypeError(f"tau_exc should be entered as a float.")
        if not isinstance(self.no_trvs_Zee, bool):
            raise TypeError(f"no_trvs_Zee should be entered as a boolean.")
        
        # Convert cm^{-1} to Hz
        self.E = wavenumber_to_Hz(self.E)
        self.Delta = wavenumber_to_Hz(self.Delta)
        self.C = wavenumber_to_Hz(self.C)
        self.D = wavenumber_to_Hz(self.D)
        self.G = wavenumber_to_Hz(self.G)
        
        # Define Omega
        self.Omega = self.C**2 * self.tau_exc
        
    def prepare_and_solve(self, **kwargs):
        """
        Prepares and solves the Liouvillian on QuTiP for a specific b and e vector.
        
        Attributes:
            **kwargs:
            b: Numpy array of floats describing vector [b_x, b_y, b_z] pointing along the applied magnetic field.
                Defaults to the value entered during object initialisation.
            e: Numpy array of floats describing vector [e_x, e_y, e_z] corresponding to the light polarisation.
                Defaults to the value entered during object initialisation.
            save: Boolean describing whether to save the QuTiP densities. Defaults to True.
            
        Returns:
            If save is set to True, then the QuTiP densities are stored as self.densities.
            If save is set to False, then the QuTiP densities are returned.
        """
        
        try:
            b = kwargs["b"]
        except KeyError:
            b = self.b
        try:
            e = kwargs["e"]
        except KeyError:
            e = self.e
        try:
            save = kwargs["save"]
        except KeyError:
            save = True
        
        print(f"Preparing the Liouvillian...")
                
        ### Extract elements of b ###
        b_x, b_y, b_z = b
        
        ### Define the coherent interactions ###
        H_mol = (self.E - self.Delta/2) * Dynamics.plus_b * Dynamics.plus_b.dag() \
            + (self.E - self.Delta/2) * Dynamics.minus_a * Dynamics.minus_a.dag() \
                + (self.E + self.Delta/2) * Dynamics.plus_a * Dynamics.plus_a.dag() \
                    + (self.E + self.Delta/2) * Dynamics.minus_b * Dynamics.minus_b.dag()
                    
        H_Zeeman_orb = qutip.Qobj([[0,0,0,0,0,0],
                                   [0,0,0,0,0,0],
                                   [0,0,1,0,0,0],
                                   [0,0,0,-1,0,0],
                                   [0,0,0,0,1,0],
                                   [0,0,0,0,0,-1]]) * self.g_L * Dynamics.mu_B * self.B * b_z
        
        H_Zeeman_spin_x = qutip.Qobj([[0,1,0,0,0,0],
                                      [1,0,0,0,0,0],
                                      [0,0,0,0,1,0],
                                      [0,0,0,0,0,1],
                                      [0,0,1,0,0,0],
                                      [0,0,0,1,0,0]]) * 0.5 * Dynamics.g_e * Dynamics.mu_B * self.B * b_x
        
        H_Zeeman_spin_y = qutip.Qobj([[0,-1j,0,0,0,0],
                                      [1j,0,0,0,0,0],
                                      [0,0,0,0,1j,0],
                                      [0,0,0,0,0,-1j],
                                      [0,0,-1j,0,0,0],
                                      [0,0,0,1j,0,0]]) * 0.5 * Dynamics.g_e * Dynamics.mu_B * self.B * b_y
        
        H_Zeeman_spin_z = qutip.Qobj([[1,0,0,0,0,0],
                                      [0,-1,0,0,0,0],
                                      [0,0,-1,0,0,0],
                                      [0,0,0,1,0,0],
                                      [0,0,0,0,1,0],
                                      [0,0,0,0,0,-1]]) * 0.5 * Dynamics.g_e * Dynamics.mu_B * self.B * b_z
        
        if self.no_trvs_Zee:
            H_Zeeman_spin_x, H_Zeeman_spin_y = 0. * H_Zeeman_spin_x, 0. * H_Zeeman_spin_y
        
        H_sb = self.G * (Dynamics.plus_a * Dynamics.minus_a.dag() \
                         + Dynamics.minus_a * Dynamics.plus_a.dag() \
                             + Dynamics.plus_b * Dynamics.minus_b.dag() \
                                 + Dynamics.minus_b * Dynamics.plus_b.dag())
        
        H = H_mol + H_Zeeman_orb + H_Zeeman_spin_x + H_Zeeman_spin_y + H_Zeeman_spin_z + H_sb
        # NOTE: The coherent part of the dynamics must be supplied in angular units
        # See https://qutip.org/docs/4.0.2/guide/dynamics/dynamics-master.html
        H = 2 * np.pi * H
        
        ### Define the dissipative processes ###
        print("*** NOTE: Only the excited-state decay is modelled here ***")               
        # Non-radiative decay
        L_decay_electronic = -1j * self.C / np.sqrt(self.Omega) * (Dynamics.zero_a * Dynamics.plus_a.dag() \
                                                                    + Dynamics.zero_a * Dynamics.minus_a.dag() \
                                                                        + Dynamics.zero_b * Dynamics.plus_b.dag() \
                                                                            + Dynamics.zero_b * Dynamics.minus_b.dag())
        L_decay_SOC = -self.D / np.sqrt(self.Omega) * (Dynamics.zero_a * Dynamics.plus_b.dag() + Dynamics.zero_b * Dynamics.minus_a.dag())
        L_decay = L_decay_electronic + L_decay_SOC
        
        ### Define the initial state ###
        if self.rho_0 is None:
            plus_dipole, minus_dipole = np.array([-1j, -1, 0]) @ e, np.array([-1j, 1, 0]) @ e
            rho_0_a = (plus_dipole * Dynamics.plus_a + minus_dipole * Dynamics.minus_a) \
                * (np.conj(plus_dipole) * Dynamics.plus_a.dag() + np.conj(minus_dipole) * Dynamics.minus_a.dag())
            rho_0_b = (plus_dipole * Dynamics.plus_b + minus_dipole * Dynamics.minus_b) \
                * (np.conj(plus_dipole) * Dynamics.plus_b.dag() + np.conj(minus_dipole) * Dynamics.minus_b.dag())
            norm = np.abs(plus_dipole)**2 + np.abs(minus_dipole)**2
            rho_0_a, rho_0_b = rho_0_a / norm, rho_0_b / norm
            rho_0 = (rho_0_a + rho_0_b) / 2
        else:
            rho_0 = self.rho_0
        
        print(f"Solving the Liouvillian...")
        
        ### Solve dynamics ###
        options = qutip.Options(method="diag")
        result = qutip.mesolve(H, rho_0, self.t_list, c_ops=[L_decay, ], options=options, progress_bar=self.progress_bar)
        densities = result.states

        if save == True:
            self.densities = densities
            print(f"The solution is stored as self.densities.")
            print("\n")
        else:
            print(f"The QuTiP densities have been returned.")
            print("\n")
            return densities
        
    def solve_g_mag(self):
        """
        Computes the ground-state magnetisation over time, stored as a NumPy array in self.g_mag.
        """
        
        print(f"Computing the ground-state magnetisation...")
        
        ### Compute magnetisation ###
        self.g_mag = np.array([Dynamics.get_g_mag(density, self.b) for density in self.densities])
        
        print(f"The ground-state magnetisation is available as self.g_mag.")
        print("\n")
    
    def plot_quantity(self, quantity, subdir, filename, figsize=(4,3), fit=False):
        """
        Plot a quantity of interest over time.
        
        Attributes:
            quantity: String describing quantity of interest.
                        Supported quantities: g_mag, spatial_ori_g_mag
            subdir: Subdirectory to which the plot should be written
            filename: Filename to which the plot should be written
            figsize: Tuple describing the figure dimensions. Defaults to (4,3).
            fit: Fit the resulting data into an exponential function.
        """
        
        print(f"Plotting {quantity}...")
        
        ### Prepare filenames ###
        subdir_full = os.path.join(os.getcwd(), subdir)
        filename_full = os.path.join(subdir_full, filename)        
        
        ### Generate plot ###
        if quantity == "g_mag":
            # Plot data
            fig, ax = plt.subplots(1, figsize = figsize, dpi = 300)
            fig_width_px = fig.get_size_inches()[0] * fig.dpi
            fig_step = max(1, len(self.t_list) // int(fig_width_px))  # Keep at most one point per pixel
            
            # Organise data into a pandas dataframe
            data = pd.DataFrame({"Time / s": self.t_list[::fig_step],
                                 "Ground-state magnetisation": self.g_mag[::fig_step]})
            
            if fit == True:
                sns.lineplot(data=data, x="Time / s", y="Ground-state magnetisation", color="black", sort=True, label="Data")
                
                tau_0 = self.Omega / self.C**2
                def g_mag_fit(t, m_0, m_1):  
                    return m_0 * np.exp(-t / tau_0) + m_1
                
                popt, _ = sp.optimize.curve_fit(g_mag_fit, data["Time / s"], data["Ground-state magnetisation"])
                m_0, m_1 = popt[0], popt[1]
                print(f"Fit complete. The fitted function is m_0 * Exp[-t / tau_0] + m_1, with m_0 = {m_0}, m_1 = {m_1}, and tau_0 = {tau_0} s.")
                
                ax.plot(data["Time / s"], g_mag_fit(data["Time / s"], *popt), color="red", label="Fit")
                ax.legend()
            
            else:
                sns.lineplot(data=data, x="Time / s", y="Ground-state magnetisation", color="black", sort=True)
            
        elif quantity == "spatial_ori_g_mag":               
            # Plot data
            fig, ax = plt.subplots(1, figsize = figsize, dpi = 300)
            fig_width_px = fig.get_size_inches()[0] * fig.dpi
            fig_step = max(1, len(self.t_list_spatial_ori) // int(fig_width_px))  # Keep at most one point per pixel
            
            # Organise data into a pandas dataframe
            data = pd.DataFrame({"Time / s": self.t_list_spatial_ori[::fig_step],
                                 "Average ground-state magnetisation": self.spatial_ori_g_mag[::fig_step]})
            
            if fit == True:
                sns.lineplot(data=data, x="Time / s", y="Average ground-state magnetisation", color="black", sort=True, label="Data")

                tau_0 = self.Omega / self.C**2
                def g_mag_fit(t, m_0, m_1):  
                    return m_0 * np.exp(-t / tau_0) + m_1
                
                popt, _ = sp.optimize.curve_fit(g_mag_fit, data["Time / s"], data["Average ground-state magnetisation"])
                m_0, m_1 = popt[0], popt[1]
                print(f"Fit complete. The fitted function is m_0 * Exp[-t / tau_0] + m_1, with m_0 = {m_0}, m_1 = {m_1}, and tau_0 = {tau_0} s.")
                
                ax.plot(data["Time / s"], g_mag_fit(data["Time / s"], *popt), color="red", label="Fit")
                ax.legend()
            
            else:
                sns.lineplot(data=data, x="Time / s", y="Average ground-state magnetisation", color="black", sort=True)
            
        else:
            raise KeyError(f"Plotting of {quantity} is not yet supported.")
            
        # Some aesthetic stuff     
        # for ax_num,ax in enumerate(axes.flat):
        # Thicken plot borders
        [i.set_linewidth(2) for i in ax.spines.values()]
        ax.tick_params(width=2)
        
        # # Plot legend
        # ax.legend()
        
        # # Set range of y-axis
        # ax.set_ylim([0.865,0.87])
        
        # # Set range of x-axis
        # ax.set_xlim([0.0,0.2])
        
        ### Save plot ###
        fig.tight_layout()
        fig.savefig(filename_full +'.jpg', bbox_inches='tight')
        plt.close()
        
        print(f"The following files were saved: ")
        print(f"\t {filename_full}.jpg")
        print("\n")
        
    def save(self, subdir, filename):
        """
        Saves itself using dill (.pkl).
        
        Attributes:
            subdir: Subdirectory to which the results should be written
            filename: Filename to which the results should be written
        """
        
        print(f"Saving...")
        
        ### Prepare filenames ###
        subdir_full = os.path.join(os.getcwd(), subdir)
        filename_full = os.path.join(subdir_full, filename)
        
        ### Dill itself ###
        with open(filename_full + ".pkl", "wb") as file:
            dill.dump(self, file)
        
        print(f"The following files were saved: ")
        print(f"\t {filename_full}.pkl")
        print("\n")

    @staticmethod
    def load(subdir, filename):
        """
        Loads results from the save file of dill format (.pkl).
        
        Usage: cell = Dynamics.load(subdir, filename)
        See last comment of https://stackoverflow.com/questions/2709800/how-to-pickle-yourself
        
        Attributes:
            subdir: Subdirectory from which the results should be loaded
            filename: Filename from which the results should be loaded
        """
        
        # Prepare filenames
        subdir_full = os.path.join(os.getcwd(), subdir)
        filename_full = os.path.join(subdir_full, filename)
        
        with open(filename_full + ".pkl", "rb") as file:
            print(f"Loading the following file: ")
            print(f"\t {filename_full}.pkl")
            print("\n")
            return dill.load(file)
        
class DynamicsSpatialOri(Dynamics):
    """
    The equivalent of a Dynamics object that is adapted for spatial averaging.
    
    Attributes:
        ** Same as Dynamics with the following additional settings: **
        
        quad_order: Lebedev quadrature order for the b vector.
                        If set to zero, the spherical coordinates of the b vector will be discretised regularly over two 1D grids
                            with num_e_points equally spaced points along phi and num_e_points/2 equally spaced points along theta.
        num_e_points: Number of equally spaced e points used per b point
        num_procs = Number of parallel processes
        
        **kwargs
        coarsening_factor: Integer describing factor over which t_list should be coarsened after the computation is complete.
                            This can reduce the storage space and time needed to compute the magnetisation without comprising on the
                            integration quality, which is set by t_list.
                            The NumPy array of timepoints over which the densities are stored (in seconds) is available as 
                            self.t_list_spatial_ori.
        theta_shift: Float describing a shift in the angle theta sampled when using the regular pair of 1D grids.
                        Defaults to zero.
        phi_shift: Float describing a shift in the angle phi sampled when using the regular pair of 1D grids.
                        Defaults to zero.
    """
    
    def __init__(self,
                 E: float,
                 Delta: float,
                 g_L: float,
                 B: float,
                 C: float,
                 D: float,
                 t_list: np.ndarray,
                 quad_order: int,
                 num_e_points: int,
                 num_procs: int,
                 **kwargs):
        
        # Inherit the parent's methods and properties
        super().__init__(E, Delta, g_L, B, C, D, t_list, **kwargs)
        
        self.quad_order = quad_order
        self.num_e_points = num_e_points
        self.num_procs = num_procs
        try:
            self.coarsening_factor = kwargs["coarsening_factor"]
        except KeyError:
            self.coarsening_factor = 1
        try:
            self.theta_shift = kwargs["theta_shift"]
        except KeyError:
            self.theta_shift = 0.
        try:
            self.phi_shift = kwargs["phi_shift"]
        except KeyError:
            self.phi_shift = 0.
            
        # Check that all inputs are valid
        if not isinstance(self.coarsening_factor, int):
            raise TypeError(f"coarsening_factor should be entered as a integer.")
        if not isinstance(self.theta_shift, float):
            raise TypeError(f"theta_shift should be entered as a float.")
        if not isinstance(self.phi_shift, float):
            raise TypeError(f"phi_shift should be entered as a float.")
        
        print(f"Calculations of different spatial orientations will be performed with {self.num_procs} parallel processes.")
        print("\n")
        
    def spatial_ori(self):
        """
        Simulate the dynamics for different spatial orientations.
        The Lebedev quadrature will be used for the b vector.
        For each b vector, we shall use 2 * (self.quad_order - 1) equally spaced e points.
        
        The points and weights of the b vector are saved as self.spatial_ori_b_points and self.spatial_ori_b_weights, respectively.
        Points of the e vector are saved as self.spatial_ori_e_points, with the element self.spatial_ori_e_points[j,k] 
            containing the vector e_j_k.
        
        The simulation results are saved as self.spatial_ori_densities.
        The element self.spatial_ori_densities[j,k] contains a QuTiP density object for the point b_j and e_j_k.
        """
        
        ### Preparing points of the b vector ###
        if self.quad_order == 0:
            ### Using regular grid ###
            print(f"The quad_order parameter was set to zero.")
            print(f"The spherical coordinates of the b vector will be discretised regularly over two 1D grids.")
            print(f"We shall choose {self.num_e_points} equally spaced points along phi and {int(self.num_e_points/2)} equally spaced points along theta.")
            
            dtheta, dphi = np.pi / int(self.num_e_points/2) / 2, 2 * np.pi / self.num_e_points / 2
            # thetas, phis = np.linspace(0 + dtheta + self.theta_shift, np.pi + dtheta + self.theta_shift, int(self.num_e_points/2), endpoint=False), np.linspace(0 + dphi + self.phi_shift, 2 * np.pi + dphi + self.phi_shift, self.num_e_points, endpoint=False)
            thetas = np.concatenate((np.linspace(0 + dtheta + self.theta_shift, np.pi / 2 + dtheta + self.theta_shift, int(self.num_e_points/4), endpoint=False),
                                     np.linspace(np.pi / 2 + dtheta - self.theta_shift, np.pi + dtheta - self.theta_shift, int(self.num_e_points/4), endpoint=False)),
                                    axis = 0)
            phis = np.linspace(0 + dphi + self.phi_shift, 2 * np.pi + dphi + self.phi_shift, self.num_e_points, endpoint=False)
            self.spatial_ori_b_points = np.array([[np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)] for theta in thetas for phi in phis])
            self.spatial_ori_b_weights = np.array([4 * np.pi**2 / self.num_e_points**2 * np.sin(theta) for theta in thetas for phi in phis])
            self.num_b_points = len(self.spatial_ori_b_weights)
            print(f"This yields {self.num_b_points} points.")   
            
        else:
            ### Extract Lebedev quadrature ###
            print(f"The Lebedev quadrature of order {self.quad_order} will be used for the b vectors.")
            self.spatial_ori_b_points, self.spatial_ori_b_weights = lebedev_rule(self.quad_order)
            self.spatial_ori_b_points = self.spatial_ori_b_points.T     # Make the points occupy rows rather than columns
            self.num_b_points = len(self.spatial_ori_b_weights)
            print(f"This order comprises {self.num_b_points} points.")
        
        
        ### Define process indices ###
        proc_indices = np.arange(0, self.num_b_points * self.num_e_points, 1, dtype=int)
        proc_map = [(j, k) for j in range(self.num_b_points) for k in range(self.num_e_points)]
        
        
        ### Compute points of the e vector ###
        print(f"{self.num_e_points} equally spaced e points will be used.")
        print(f"Computing points of the e vector...")
        
        # # Generate the rotation matrix that takes the initial b vector to initial e vector
        # R_b_e = get_rot_mat_b_to_a(self.e, self.b)
        
        # Define parallel operation
        def operation(proc_index):
            # Extract parameters
            j, k = proc_map[proc_index]
            b_j = self.spatial_ori_b_points[j]
            
            # # Find the new e vector for this b_j point
            # e_j = R_b_e @ b_j
            
            # Generate the rotation matrix that takes the initial b vector to this b vector
            R_b_b_j = get_rot_mat_b_to_a(b_j, self.b)
            
            # Find the new e vector for this b_j point
            e_j = R_b_b_j @ self.e
            
            # Generate the rotation matrix that rotates about b_j
            R_b_j = get_rot_mat_about_a(b_j, self.num_e_points)
            
            # Rotate the e_j vector
            e_j_k = np.linalg.matrix_power(R_b_j, k) @ e_j
            
            sys.stdout.flush()      # Flush all output from the process
            return e_j_k
        
        # Run parallel operations
        procs_pool = mp.Pool(self.num_procs)
        e_j_ks = procs_pool.map(operation, proc_indices)
        procs_pool.close()  # Close the pool
        procs_pool.terminate()  # Kill the pool
        
        # Organise data
        self.spatial_ori_e_points = np.empty((self.num_b_points, self.num_e_points), dtype=np.ndarray)
        for proc_index, (j,k) in enumerate(proc_map):
            self.spatial_ori_e_points[j,k] = e_j_ks[proc_index]
        
        print(f"Done.")
        print("\n")
        
        
        ### Run simulation for each point ###
        # Define parallel operation
        def operation(proc_index):
            # Extract parameters
            j, k = proc_map[proc_index]
            b_j, e_j_k = self.spatial_ori_b_points[j], self.spatial_ori_e_points[j,k]
            
            print(f"Running b point {j}: {b_j}, and e point {k}: {e_j_k}...\n", end="")
            # (Explicitly including the linebreak at the end of the print helps prevent prints from interrputing each other)
            # See https://stackoverflow.com/a/69116755
            
            # Solve dynamics
            with HiddenPrints():
                densities = self.prepare_and_solve(b=b_j, e=e_j_k, save=False)
                
                # Coarsen time domain, if requested
                if self.coarsening_factor != 1:
                    print(f"Coarsening of the time domain is requested. Only every {self.coarsening_factor} element will be kept.")
                    densities = np.array(densities)[::self.coarsening_factor]
            
            print(f"Completed b point {j}: {b_j}, and e point {k}: {e_j_k}.\n", end="")
            
            sys.stdout.flush()      # Flush all output from the process
            return densities
        
        # Run parallel operations
        procs_pool = mp.Pool(self.num_procs)
        densities_j_ks = procs_pool.map(operation, proc_indices)
        procs_pool.close()  # Close the pool
        procs_pool.terminate()  # Kill the pool
        
        # Coarsen time domain, if requested
        if self.coarsening_factor != 1:
            self.t_list_spatial_ori = self.t_list[::self.coarsening_factor]
        else:
            self.t_list_spatial_ori = self.t_list
        
        # Organise data
        self.spatial_ori_densities = np.empty((self.num_b_points, self.num_e_points), dtype=qutip.qobj.Qobj)
        for proc_index, (j,k) in enumerate(proc_map):
            self.spatial_ori_densities[j,k] = densities_j_ks[proc_index]
        
        print(f"All {self.num_b_points * self.num_e_points} simulations are now complete.")
        print(f"The points and weights of the b vector are saved as self.spatial_ori_b_points and self.spatial_ori_b_weights, respectively.")
        print(f"Points of the e vector are saved as self.spatial_ori_e_points, with the element self.spatial_ori_e_points[j,k] containing the vector e_j_k.")
        print(f"The simulation results are saved as self.spatial_ori_densities.")
        print(f"The element self.spatial_ori_densities[j,k] contains a QuTiP density object for the point b_j and e_j_k.")
        print("\n")
        
    def spatial_ori_solve_g_mag(self, method=1):
        """
        Computes the spatially averaged ground-state magnetisation over time, stored as a NumPy array in self.spatial_ori_g_mag.
        The results from self.spatial_ori_densities will be used.
        
        Attributes:
            method: Method for parallelising the computation.
                        1 = Computes the magnetisation for all timepoints before summing
                        2 = Sums the magnetisation for each timepoint on-the-fly (less memory-intensive)
                    Defaults to 1.
        """
        
        print(f"Computing the spatially averaged ground-state magnetisation...")
        
        if method == 1:
            ### Define process indices ###
            proc_indices = np.arange(0, self.num_b_points * self.num_e_points, 1, dtype=int)
            proc_map = [(j, k) for j in range(self.num_b_points) for k in range(self.num_e_points)]
            
            ### Compute magnetisation for each point ###
            # Define parallel operation
            def operation(proc_index):
                # Extract parameters
                j, k = proc_map[proc_index]
                w_j = self.spatial_ori_b_weights[j]
                b_j = self.spatial_ori_b_points[j]
                densities = self.spatial_ori_densities[j,k]
                
                # Compute weighted magnetisation
                weight = w_j / (4 * np.pi * self.num_e_points)
                g_mag = weight * np.array([Dynamics.get_g_mag(density, b_j) for density in densities])
                
                sys.stdout.flush()      # Flush all output from the process
                return g_mag
                
            # Run parallel operations
            procs_pool = mp.Pool(self.num_procs)
            g_mags = procs_pool.map(operation, proc_indices)
            procs_pool.close()  # Close the pool
            procs_pool.terminate()  # Kill the pool
            
            ### Sum up all weighted magnetisations ###
            g_mags = np.array(g_mags)   # Make NumPy array
            
            # Define parallel operation
            def operation(proc_index):
                return np.sum(g_mags[:,proc_index])
            
            # Run parallel operations
            procs_pool = mp.Pool(self.num_procs)
            self.spatial_ori_g_mag = procs_pool.map(operation, range(len(self.t_list_spatial_ori)))
            procs_pool.close()  # Close the pool
            procs_pool.terminate()  # Kill the pool
        
        elif method == 2:
            ###########################################################################################
            # A less memory-intensive method, achieved by summing the magnetisation at each timepoint #
            # rather than storing the magnetisation for all timepoints before summing.                #
            ###########################################################################################
            
            print("Using a less memory-intensive method...")
            
            ### Define process indices ###
            proc_indices = np.arange(0, len(self.t_list_spatial_ori), 1, dtype=int)
            jk_indices = [(j, k) for j in range(self.num_b_points) for k in range(self.num_e_points)]
            
            ### Compute magnetisation for each timepoint ###
            # Define parallel operation
            def operation(proc_index):
                density_index = proc_index
                
                g_mag = 0
                for j, k in jk_indices:
                    # Extract parameters
                    w_j = self.spatial_ori_b_weights[j]
                    b_j = self.spatial_ori_b_points[j]
                    density = self.spatial_ori_densities[j,k][density_index]
                    
                    # Compute weighted magnetisation
                    weight = w_j / (4 * np.pi * self.num_e_points)
                    g_mag += weight * Dynamics.get_g_mag(density, b_j)
                
                sys.stdout.flush()      # Flush all output from the process
                return g_mag 
            
            # Run parallel operations
            procs_pool = mp.Pool(self.num_procs)
            self.spatial_ori_g_mag = procs_pool.map(operation, proc_indices)
            procs_pool.close()  # Close the pool
            procs_pool.terminate()  # Kill the pool
            
        else:
            raise NotImplementedError(f"The chosen method is invalid. Chosen method: {method}")
        
        self.spatial_ori_g_mag = np.array(self.spatial_ori_g_mag)   # Make NumPy array

        print(f"The spatially averaged ground-state magnetisation is available as self.spatial_ori_g_mag.")
        print("\n")
    