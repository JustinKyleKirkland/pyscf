#!/usr/bin/env python
# Copyright 2014-2026 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''
State-specific non-equilibrium solvation for the PCM family.

A vertical process (ionization, electron attachment, a Delta-SCF excitation)
is fast compared with the reorientation of the solvent molecules. Only the
electronic (fast) polarization of the solvent follows it; the orientational
(slow) polarization stays frozen in the arrangement it had for the initial
state.

Splitting the apparent surface charge of the initial state with the Pekar
partition,

    q_slow = q[rho_ref; eps] - q[rho_ref; eps_optical]

leaves a frozen slow charge that enters the Hamiltonian of the final state as a
fixed one-electron operator, while the fast charge is solved self-consistently
at the optical dielectric constant. The decomposition is exact for C-PCM and
COSMO and well defined for IEF-PCM and SS(V)PE.

Unlike ``PCM.equilibrium_solvation``, which handles the linear-response
treatment of excited states, this module applies to state-specific processes
where the final state is obtained from its own SCF.

Energies only; nuclear gradients are not implemented.

References:
    Tomasi, Mennucci, Cammi, Chem. Rev. 105, 2999 (2005)
    Scalmani, Frisch, J. Chem. Phys. 132, 114110 (2010)
    Lange, Herbert, J. Chem. Phys. 133, 244111 (2010)
    Marenich et al., Chem. Sci. 2, 2143 (2011)
'''

import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.solvent import pcm as _pcm
from pyscf.solvent import _attach_solvent


class SlowPolarization(lib.StreamObject):
    '''Frozen slow polarization of the solvent, taken from a reference state.

    Attributes:
        v_slow : (nao,nao) ndarray
            One-electron operator produced by the frozen slow surface charges.
        q_slow : (ngrids,) ndarray
            The frozen slow apparent surface charges.
        e_slow : float
            Constant energy offset. It restores the nuclear interaction with
            the slow charges, which the electron-only operator drops, and
            carries the one-half factor of the linear slow term.
        eps, eps_optical : float
            Static and optical dielectric constants.
        method : str
            PCM family of the reference calculation.
        surface : dict
            Cavity of the reference calculation, used to check that the final
            state is set up on the same cavity.
    '''

    def __init__(self, v_slow, q_slow, e_slow, eps, eps_optical, method,
                 surface):
        self.v_slow = v_slow
        self.q_slow = q_slow
        self.e_slow = e_slow
        self.eps = eps
        self.eps_optical = eps_optical
        self.method = method
        self.surface = surface

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('** Frozen slow polarization **')
        log.info('method = %s', self.method)
        log.info('eps = %g, eps_optical = %g', self.eps, self.eps_optical)
        log.info('ngrids = %d, sum(q_slow) = %.6f, e_slow = %.9g',
                 self.q_slow.size, self.q_slow.sum(), self.e_slow)
        return self


def build_slow_polarization(mf_ref, eps_optical=None, dm=None):
    '''Pekar partition of the solvent response of a converged reference state.

    Args:
        mf_ref : an SCF object carrying a converged pcm.PCM solvent

    Kwargs:
        eps_optical : float
            Optical dielectric constant. Defaults to the value the reference
            solvent supplies, which comes from the solvent name when one was
            given.
        dm : ndarray
            Density matrix defining the slow polarization. Defaults to the
            density of the reference calculation.

    Returns:
        A :class:`SlowPolarization` instance.
    '''
    pcmobj = getattr(mf_ref, 'with_solvent', None)
    if pcmobj is None:
        raise RuntimeError(
            'The reference calculation carries no solvent. Run something like '
            'mf = mol.RHF().PCM("water"); mf.run() first.')
    if not isinstance(pcmobj, _pcm.PCM):
        raise NotImplementedError(
            f'Non-equilibrium solvation for {type(pcmobj).__name__}. Only the '
            'PCM family implements the Pekar partition.')
    if not mf_ref.converged:
        logger.warn(mf_ref, 'Reference state is not converged. The frozen slow '
                    'polarization will be built from an unconverged density.')

    if eps_optical is None:
        eps_optical = pcmobj.get_eps_optical()
    eps_optical = float(eps_optical)
    if eps_optical > pcmobj.eps:
        raise ValueError(
            f'eps_optical={eps_optical} exceeds eps={pcmobj.eps}. The optical '
            'constant describes only the fast part of the same response and '
            'cannot be the larger of the two.')

    if dm is None:
        dm = mf_ref.make_rdm1()
    dm = numpy.asarray(dm)
    if dm.ndim == 3:
        dm = dm[0] + dm[1]

    # Full response of the reference density at the static dielectric constant.
    pcmobj._get_vind(dm)
    q_full = pcmobj._intermediates['q_sym'].copy()
    v_grids = pcmobj._intermediates['v_grids'].copy()
    v_grids_n = pcmobj.v_grids_n.copy()

    # Same cavity, same density, optical dielectric constant.
    pcm_fast = _clone_at_eps(pcmobj, eps_optical)
    pcm_fast._get_vind(dm)
    q_fast = pcm_fast._intermediates['q_sym'].copy()

    q_slow = q_full - q_fast
    v_slow = pcm_fast._get_vmat(q_slow)[0].copy()

    # The operator above acts on the electrons only. The first term restores
    # the nuclear interaction with the slow charges; the second is the one-half
    # factor carried by the linear slow term.
    e_slow = float(numpy.dot(q_slow, v_grids_n)
                   - 0.5 * numpy.dot(q_slow, v_grids))

    return SlowPolarization(v_slow, q_slow, e_slow, pcmobj.eps, eps_optical,
                            pcmobj.method, pcmobj.surface)


def _clone_at_eps(pcmobj, eps):
    '''A PCM on the same cavity and with the same settings, at another eps.'''
    new = _pcm.PCM(pcmobj.mol)
    new.method = pcmobj.method
    new.eps = eps
    new.lebedev_order = pcmobj.lebedev_order
    new.vdw_scale = pcmobj.vdw_scale
    new.r_probe = pcmobj.r_probe
    new.radii_table = pcmobj.radii_table
    new.surface_discretization_method = pcmobj.surface_discretization_method
    new.max_memory = pcmobj.max_memory
    new.verbose = pcmobj.verbose
    new.build()
    _check_same_cavity(new, pcmobj.surface)
    return new


def _check_same_cavity(pcmobj, surface):
    '''The slow charges live on the reference cavity. A cavity built for the
    final state has to be the same one, or the two sets of surface charges
    refer to different points in space.
    '''
    ref = surface['grid_coords']
    new = pcmobj.surface['grid_coords']
    if new.shape != ref.shape or abs(new - ref).max() > 1e-10:
        raise RuntimeError(
            'The cavity of the final state differs from the cavity the frozen '
            'slow polarization was built on. A vertical process must keep the '
            'geometry and the cavity settings of the reference state.')


class _NonEquilibriumSCF(_attach_solvent._Solvation):
    '''Adds the frozen slow operator to hcore and its constant to the energy.'''

    def get_hcore(self, mol=None):
        return super().get_hcore(mol) + self.slow_polarization.v_slow

    def energy_elec(self, dm=None, h1e=None, vhf=None):
        e_tot, e_coul = super().energy_elec(dm, h1e, vhf)
        e_tot = e_tot + self.slow_polarization.e_slow
        if getattr(self, 'scf_summary', None) is not None:
            self.scf_summary['e_slow'] = self.slow_polarization.e_slow
        return e_tot, e_coul

    def dump_flags(self, verbose=None):
        super().dump_flags(verbose)
        self.slow_polarization.dump_flags(verbose)
        return self

    def nuc_grad_method(self):
        raise NotImplementedError(
            'Nuclear gradients for state-specific non-equilibrium solvation')

    Gradients = nuc_grad_method


def for_scf(mf, slow_polarization):
    '''Attach a frozen slow polarization to an SCF object.

    The fast polarization is handled by an ordinary PCM at the optical
    dielectric constant, so it relaxes together with the density during the
    SCF, while the slow part stays fixed.

    Args:
        mf : an SCF object for the final state
        slow_polarization : :class:`SlowPolarization`

    Returns:
        The decorated SCF object.
    '''
    if isinstance(mf, _NonEquilibriumSCF):
        mf.slow_polarization = slow_polarization
        return mf

    fast = _pcm.PCM(mf.mol)
    fast.method = slow_polarization.method
    fast.eps = slow_polarization.eps_optical
    fast.build()
    _check_same_cavity(fast, slow_polarization.surface)

    nao = mf.mol.nao
    if slow_polarization.v_slow.shape != (nao, nao):
        raise RuntimeError(
            f'The frozen slow operator has shape '
            f'{slow_polarization.v_slow.shape} but the final state has '
            f'nao={nao}. Both states must use the same basis.')

    sol_mf = _attach_solvent._for_scf(mf, fast)
    sol_mf.slow_polarization = slow_polarization
    name = 'NonEquilibrium' + sol_mf.__class__.__name__
    return lib.set_class(sol_mf, (_NonEquilibriumSCF, sol_mf.__class__), name)


def nonequilibrium(mf, mf_ref, eps_optical=None, dm=None):
    '''Solve an SCF with the slow solvent polarization frozen at a reference.

    This is the state-specific counterpart of ``PCM.equilibrium_solvation``:
    the reference state fixes the orientational polarization, and the final
    state relaxes only the electronic part of the solvent response.

        >>> mf_ref = mol.RHF().PCM('water').run()
        >>> cation = mol.copy()
        >>> cation.charge, cation.spin = 1, 1
        >>> cation.build(False, False)
        >>> mf = solvent.nonequilibrium(cation.UHF(), mf_ref).run()

    Args:
        mf : an SCF object for the final state, on the same geometry and basis
        mf_ref : a converged SCF carrying the reference PCM solvent

    Kwargs:
        eps_optical : float
            Optical dielectric constant, otherwise taken from the reference.
        dm : ndarray
            Density defining the slow polarization, otherwise the reference's.

    Returns:
        The decorated SCF object.
    '''
    return for_scf(mf, build_slow_polarization(mf_ref, eps_optical, dm))
