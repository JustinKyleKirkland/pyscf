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

import unittest
import numpy
from pyscf import gto, scf
from pyscf.solvent import pcm, noneq


def setUpModule():
    global mol
    mol = gto.M(atom='O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587',
                basis='sto-3g', verbose=0)

def tearDownModule():
    global mol
    del mol


def _pcm_rhf(m, eps=78.3553, method='C-PCM'):
    cm = pcm.PCM(m)
    cm.method = method
    cm.eps = eps
    cm.lebedev_order = 29
    mf = scf.RHF(m).PCM(cm)
    mf.conv_tol = 1e-12
    mf.kernel()
    return mf

def _cation(m):
    c = m.copy()
    c.charge += 1
    c.spin = 1
    c.build(False, False)
    return c


class KnownValues(unittest.TestCase):
    def test_equilibrium_limit(self):
        # With no separation of timescales the frozen slow polarization must
        # collapse onto the ordinary equilibrium solvent.
        ref = _pcm_rhf(mol)
        mf = noneq.nonequilibrium(scf.RHF(mol), ref,
                                  eps_optical=ref.with_solvent.eps)
        mf.conv_tol = 1e-12
        self.assertAlmostEqual(mf.kernel(), ref.e_tot, 9)

    def test_gas_phase_limit(self):
        # eps = eps_optical = 1 is no solvent at all.
        gas = scf.RHF(mol)
        gas.conv_tol = 1e-12
        gas.kernel()
        ref = _pcm_rhf(mol, eps=1.0)
        mf = noneq.nonequilibrium(scf.RHF(mol), ref, eps_optical=1.0)
        mf.conv_tol = 1e-12
        self.assertAlmostEqual(mf.kernel(), gas.e_tot, 9)

    def test_pekar_split_conserves_charge(self):
        # q_slow is defined as the difference of two surface charge solutions,
        # so the three sets have to add up exactly.
        ref = _pcm_rhf(mol)
        slow = noneq.build_slow_polarization(ref)
        dm = ref.make_rdm1()
        smod = ref.with_solvent
        smod._get_vind(dm)
        q_full = smod._intermediates['q_sym'].copy()
        fast = noneq._clone_at_eps(smod, slow.eps_optical)
        fast._get_vind(dm)
        q_fast = fast._intermediates['q_sym']
        self.assertAlmostEqual(abs(slow.q_slow + q_fast - q_full).max(), 0, 12)

    def test_ionization_ordering(self):
        # Equilibrium solvent lets the slow polarization relax onto the cation
        # and over-stabilizes it, so it puts the ionization energy below the
        # non-equilibrium value. Gas phase has no stabilization at all.
        ref = _pcm_rhf(mol)
        cat = _cation(mol)

        gas_n = scf.RHF(mol); gas_n.conv_tol = 1e-11; gas_n.kernel()
        gas_c = scf.UHF(cat); gas_c.conv_tol = 1e-11; gas_c.kernel()
        vie_gas = gas_c.e_tot - gas_n.e_tot

        eq_c = scf.UHF(cat).PCM(pcm.PCM(cat))
        eq_c.with_solvent.eps = 78.3553
        eq_c.with_solvent.lebedev_order = 29
        eq_c.conv_tol = 1e-11
        eq_c.kernel()
        vie_eq = eq_c.e_tot - ref.e_tot

        ne_c = noneq.nonequilibrium(scf.UHF(cat), ref)
        ne_c.conv_tol = 1e-11
        ne_c.kernel()
        vie_ne = ne_c.e_tot - ref.e_tot

        self.assertTrue(vie_eq < vie_ne < vie_gas)

    def test_anion_reference(self):
        # The slow polarization is built from whatever reference is given, so a
        # charged reference has to work as well as a neutral one.
        anion = gto.M(atom='O 0 0 0; H 0 0 0.97', basis='sto-3g',
                      charge=-1, verbose=0)
        ref = _pcm_rhf(anion)
        neutral = anion.copy()
        neutral.charge = 0
        neutral.spin = 1
        neutral.build(False, False)
        mf = noneq.nonequilibrium(scf.UHF(neutral), ref)
        mf.conv_tol = 1e-11
        mf.kernel()
        self.assertTrue(mf.converged)

        # The slow polarization is organised around a charge that is no longer
        # there, which costs the neutral product energy relative to letting the
        # solvent relax. That comparison is basis independent, unlike the sign
        # of the detachment energy itself, which a minimal basis gets wrong.
        eq = scf.UHF(neutral).PCM(pcm.PCM(neutral))
        eq.with_solvent.eps = 78.3553
        eq.with_solvent.lebedev_order = 29
        eq.conv_tol = 1e-11
        eq.kernel()
        self.assertTrue(mf.e_tot > eq.e_tot)

    def test_slow_polarization_is_frozen(self):
        # The stored operator must not change when the final state relaxes.
        ref = _pcm_rhf(mol)
        slow = noneq.build_slow_polarization(ref)
        before = slow.v_slow.copy()
        mf = noneq.for_scf(scf.UHF(_cation(mol)), slow)
        mf.conv_tol = 1e-11
        mf.kernel()
        self.assertAlmostEqual(abs(slow.v_slow - before).max(), 0, 12)

    def test_cavity_mismatch_raises(self):
        ref = _pcm_rhf(mol)
        moved = gto.M(atom='O 0 0 0; H 0 -0.80 0.60; H 0 0.80 0.60',
                      basis='sto-3g', verbose=0)
        slow = noneq.build_slow_polarization(ref)
        self.assertRaises(RuntimeError, noneq.for_scf, scf.RHF(moved), slow)

    def test_basis_mismatch_raises(self):
        ref = _pcm_rhf(mol)
        slow = noneq.build_slow_polarization(ref)
        big = gto.M(atom=mol.atom, basis='6-31g', verbose=0)
        # a different basis on the same geometry keeps the cavity but breaks
        # the shape of the frozen operator
        self.assertRaises(RuntimeError, noneq.for_scf, scf.RHF(big), slow)

    def test_requires_pcm(self):
        bare = scf.RHF(mol)
        bare.conv_tol = 1e-10
        bare.kernel()
        self.assertRaises(RuntimeError, noneq.build_slow_polarization, bare)

        dd = scf.RHF(mol).DDCOSMO()
        dd.conv_tol = 1e-10
        dd.kernel()
        self.assertRaises(NotImplementedError,
                          noneq.build_slow_polarization, dd)

    def test_all_pcm_families(self):
        # COSMO, IEF-PCM and SS(V)PE are method strings on the same class, so
        # the Pekar split applies to all of them, not only C-PCM.
        cat = _cation(mol)
        for method in ('C-PCM', 'COSMO', 'IEF-PCM', 'SS(V)PE'):
            ref = _pcm_rhf(mol, method=method)
            mf = noneq.nonequilibrium(scf.UHF(cat), ref)
            mf.conv_tol = 1e-11
            mf.kernel()
            self.assertTrue(mf.converged)
            self.assertEqual(mf.slow_polarization.method, method)

    def test_smd_refused(self):
        # SMD subclasses PCM, so it would pass an isinstance check. Its CDS
        # term cannot be carried into the final state, and leaving it on one
        # side only is an error the size of the effect being computed.
        from pyscf.solvent import smd
        sm = smd.SMD(mol)
        sm.solvent = 'water'
        fake = scf.RHF(mol)
        fake.with_solvent = sm
        fake.converged = True
        self.assertRaises(NotImplementedError,
                          noneq.build_slow_polarization, fake)

    def test_eps_none_refused(self):
        bare = _pcm_rhf(mol)
        bare.with_solvent.eps = None
        self.assertRaises(RuntimeError, noneq.build_slow_polarization, bare)

    def test_eps_optical_bounds(self):
        ref = _pcm_rhf(mol)
        self.assertRaises(ValueError, noneq.build_slow_polarization,
                          ref, 1e6)

    def test_gradients_not_implemented(self):
        ref = _pcm_rhf(mol)
        mf = noneq.nonequilibrium(scf.UHF(_cation(mol)), ref)
        mf.conv_tol = 1e-10
        mf.kernel()
        self.assertRaises(NotImplementedError, mf.nuc_grad_method)


if __name__ == '__main__':
    print('Full Tests for state-specific non-equilibrium solvation')
    unittest.main()
