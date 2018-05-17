# Import modules
import numpy as np
import pyscf.gto
import pyscf.scf

from numpy import linalg
# Specify molecular geometry and basis set
mol = pyscf.gto.M(
verbose = 5,
atom = [
['O', (-1.81415925, 1.62241885, 0.0)],
['H', (-0.85415925, 1.62241885, 0.0)],
['H', (-2.13461384, 2.52735469, 0.0)],
],
basis = 'sto-3G',
symmetry = True)

# Set options to make Numpy printing more clear
np.set_printoptions(linewidth=150, edgeitems=10, suppress=True)
# Obtain the number of atomic orbitals in the basis set
nao = mol.nao_nr()
# Obtain the number of electrons
nelec = mol.nelectron
# Compute nuclear repulsion energy
enuc = mol.energy_nuc()
# Compute overlap integrals
ovlp = mol.intor('cint1e_ovlp_sph')
# Compute one-electron kinetic integrals
T = mol.intor('cint1e_kin_sph')
# Compute one-electron potential integrals
V = mol.intor('cint1e_nuc_sph')
# Compute two-electron repulsion integrals (Chemists' notation)
v2e = mol.intor('cint2e_sph').reshape((nao,)*4)

#    Diagonalize overlap integral martrix
S_eval, S_evec = np.linalg.eigh(ovlp)
S_root_half = np.diagflat(1/(np.sqrt(S_eval)))
trans_S = np.linalg.multi_dot([S_evec,S_root_half,np.transpose(S_evec)])

#    Set up initial Fock matrix
initial_Fock = T + V

#    Compute transformed Fock matrix
trans_Fock = np.linalg.multi_dot([trans_S,initial_Fock,trans_S])

#    Compute the SCF eigenvector matrix
Fock_eval, Fock_evec = np.linalg.eigh(trans_Fock)
scf_evec = np.dot(trans_S,Fock_evec)

#    Compute initial density matrix
n = nelec/2
initial_dens = np.dot(scf_evec[:,:n],np.transpose(scf_evec[:,:n]))
Eold = 0.00

#    Starting SCF
for k in range(0,100):
	twoelecint = 2*np.einsum('rs,pqrs->pq',initial_dens,v2e) - np.einsum('rs,psrq->pq',initial_dens,v2e)
	tot_Fock = initial_Fock + twoelecint
	Eelec = np.einsum('pq,pq',initial_dens,initial_Fock) + np.einsum('pq,pq',initial_dens,tot_Fock)
	Etot = Eelec + enuc
	trans_Fock_new = np.linalg.multi_dot([trans_S,tot_Fock,trans_S])
	Fock_eval_new, Fock_evec_new = np.linalg.eigh(trans_Fock_new)
	scf_evec_new = np.dot(trans_S,Fock_evec_new)
	dens_new = np.dot(scf_evec_new[:,:n],np.transpose(scf_evec_new[:,:n]))

	i,j = initial_dens.shape
	sum_diff = 0
	for a in range (0,i):
		for b in range (0,j):
			diff_dens = dens_new[a][b] - initial_dens[a][b]
			sum_diff = sum_diff + diff_dens*diff_dens

	rms_dens = np.sqrt(sum_diff)
	if rms_dens <= 0.00000001 and abs(Eold-Eelec) <= 0.00000001 :
		print "SCF converged"
		print ("Energy is : "+str(Etot))
		break
	else:
		initial_dens = dens_new
		Eold = Eelec
		print ("Cycle number "+str(k))
		print ("energy "+str(Etot))

#print scf_evec_new      #Coefficient of MO in AO basis
#print Fock_eval_new     #MO energies

#      Transform the 1 electron integral to MO basis
oneelecint_mo = np.einsum('ab,ac,cd->bd',scf_evec_new,initial_Fock,scf_evec_new)

#      Transform 2 electron integral to MO Basis
twoelecint_1 = np.einsum('zs,wxyz->wxys',scf_evec_new,v2e)
twoelecint_2 = np.einsum('yr,wxys->wxrs',scf_evec_new,twoelecint_1)
twoelecint_3 = np.einsum('xq,wxrs->wqrs',scf_evec_new,twoelecint_2)
twoelecint_mo = np.einsum('wp,wqrs->pqrs',scf_evec_new,twoelecint_3)

#      Verify integrals
E_scf_mo_1 = 0
E_scf_mo_2 = 0
E_scf_mo_3 = 0
for i in range(0,n):
	E_scf_mo_1 += oneelecint_mo[i][i]
		
for i in range(0,n):
	for j in range(0,n):
		E_scf_mo_2 += 2*twoelecint_mo[i][i][j][j] - twoelecint_mo[i][j][i][j]

Escf_mo = 2*E_scf_mo_1 + E_scf_mo_2 + E_scf_mo_3 + enuc

if abs(Escf_mo - Etot)<= 1E-6 :
	print "MO conversion successful"

#      Compute MP2 energy
denominator = 0
numerator = 0
summ = 0
for i in range(0,n):
	for j in range(0,n):
		for a in range(n,nao):
			for b in range(n,nao):
				denominator = Fock_eval_new[i] + Fock_eval_new[j] - Fock_eval_new[a] - Fock_eval_new[b]
				numerator = twoelecint_mo[i,a,j,b] * (2*twoelecint_mo[i,a,j,b] - twoelecint_mo[i,b,j,a])
				val = numerator/denominator
				summ = summ + val
E_mp2 = summ
print E_mp2

print "MP2 energy : " + str(Etot+E_mp2)
