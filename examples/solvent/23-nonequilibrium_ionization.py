#!/usr/bin/env python
'''
State-specific non-equilibrium solvation.

A vertical process is fast compared with the reorientation of the solvent.
Only the electronic polarization of the solvent follows it; the orientational
part stays frozen in the arrangement it had for the initial state.

pyscf.solvent.nonequilibrium freezes the slow polarization at a converged
reference state and lets the fast part relax with the density of the final
state. Use it for vertical ionization, vertical detachment, and Delta-SCF
excitations. For linear-response excited states, use the equilibrium_solvation
flag of the PCM object instead.
'''

from pyscf import gto, scf, solvent

HARTREE2EV = 27.211386245988

mol = gto.M(atom='''
O   0.000000   0.000000   0.117790
H   0.000000   0.755453  -0.471161
H   0.000000  -0.755453  -0.471161''', basis='6-31g*', verbose=0)

# The reference state, fully equilibrated with the solvent.
mf = mol.RHF().PCM()
mf.with_solvent.eps = 78.3553
mf.with_solvent.eps_optical = 1.78
mf.run()

cation = mol.copy()
cation.charge, cation.spin = 1, 1
cation.build(False, False)

#
# Equilibrium solvent on both states lets the slow polarization relax onto the
# cation, which over-stabilizes it and pushes the ionization energy too low.
#
mf_eq = cation.UHF().PCM()
mf_eq.with_solvent.eps = 78.3553
mf_eq.run()
print('equilibrium solvent  VIE = %.3f eV'
      % ((mf_eq.e_tot - mf.e_tot) * HARTREE2EV))

#
# Freezing the slow polarization at the neutral is the physically correct
# treatment of a vertical process.
#
mf_ne = solvent.nonequilibrium(cation.UHF(), mf).run()
print('non-equilibrium      VIE = %.3f eV'
      % ((mf_ne.e_tot - mf.e_tot) * HARTREE2EV))

#
# Any charge state works as the reference, so vertical detachment energies of
# anions are set up the same way.
#
anion = gto.M(atom='O 0 0 0; H 0 0 0.97', basis='6-31+g*', charge=-1, verbose=0)
mf_anion = anion.RHF().PCM()
mf_anion.with_solvent.eps = 78.3553
mf_anion.run()

neutral = anion.copy()
neutral.charge, neutral.spin = 0, 1
neutral.build(False, False)

mf_vde = solvent.nonequilibrium(neutral.UHF(), mf_anion).run()
print('non-equilibrium      VDE = %.3f eV'
      % ((mf_vde.e_tot - mf_anion.e_tot) * HARTREE2EV))
