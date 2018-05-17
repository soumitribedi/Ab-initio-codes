# Import modules
import numpy as np
import pyscf.gto
import pyscf.scf
import copy as cp

from numpy import linalg
# Specify molecular geometry and basis set
mol = pyscf.gto.M(
verbose = 5,
atom = [
['N', (0.0, 0.0, 0.0)],
['N', (1.1, 0.0, 0.0)],
],
basis = 'sto-3g',
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

#     Setup t2 and D
occ=n
virt=nao-n
twoelecint_mo = np.swapaxes(twoelecint_mo,1,2)  #physicist notation
t2 = np.zeros((occ,occ,virt,virt))
D = np.zeros((occ,occ,virt,virt))
for i in range(0,occ):
	for j in range(0,occ):
		for a in range(occ,nao):
			for b in range(occ,nao):
				D[i,j,a-occ,b-occ] = Fock_eval_new[i] + Fock_eval_new[j] - Fock_eval_new[a] - Fock_eval_new[b]
				t2[i,j,a-occ,b-occ] = twoelecint_mo[i,j,a,b]/D[i,j,a-occ,b-occ]

#      Calculate MP2 energy
E_mp2 = 2*np.einsum('ijab,ijab',t2,twoelecint_mo[:occ,:occ,occ:nao,occ:nao]) - np.einsum('ijab,ijba',t2,twoelecint_mo[:occ,:occ,occ:nao,occ:nao])
print "MP2 correlation energy is : "+str(E_mp2)
E_mp2_tot = Etot + E_mp2
print "MP2 energy is : "+str(E_mp2_tot)
E_old = E_mp2

#      CCD begins
for x in range(0,50):
	#      Fvv
	F_vv = -2*np.einsum('mnaf,mnef->ae',t2,twoelecint_mo[:occ,:occ,occ:nao,occ:nao]) + np.einsum('mnaf,nmef->ae',t2,twoelecint_mo[:occ,:occ,occ:nao,occ:nao])
	#      Foo
	F_oo = 2*np.einsum('inef,mnef->mi',t2,twoelecint_mo[:occ,:occ,occ:nao,occ:nao]) - np.einsum('inef,mnfe->mi',t2,twoelecint_mo[:occ,:occ,occ:nao,occ:nao])
	#      Woooo
	W_oooo = twoelecint_mo[:occ,:occ,:occ,:occ] + 0.5*np.einsum('ijef,mnef->mnij',t2,twoelecint_mo[:occ,:occ,occ:nao,occ:nao])
	#      Wvvvv
	W_vvvv = twoelecint_mo[occ:nao,occ:nao,occ:nao,occ:nao] + 0.5*np.einsum('mnab,mnef->abef',t2,twoelecint_mo[:occ,:occ,occ:nao,occ:nao])
	#      Wovvo
	W_ovvo = twoelecint_mo[:occ,occ:nao,occ:nao,:occ] - 0.5*np.einsum('jnfb,mnef->mbej',t2,twoelecint_mo[:occ,:occ,occ:nao,occ:nao]) + np.einsum('njfb,mnef->mbej',t2,twoelecint_mo[:occ,:occ,occ:nao,occ:nao]) - 0.5*np.einsum('njfb,nmef->mbej',t2,twoelecint_mo[:occ,:occ,occ:nao,occ:nao])
	#      Wovov
	W_ovov = -twoelecint_mo[:occ,occ:nao,:occ,occ:nao] + 0.5*np.einsum('jnfb,nmef->mbje',t2,twoelecint_mo[:occ,:occ,occ:nao,occ:nao])

	#      Compute R_ijab
	#R_ijab = twoelecint_mo[:occ,:occ,occ:nao,occ:nao]
	R_ijab = cp.deepcopy(twoelecint_mo[:occ,:occ,occ:nao,occ:nao])
	R_ijab += np.einsum('ijae,be->ijab',t2,F_vv)
	R_ijab += np.einsum('jibe,ae->ijab',t2,F_vv)
	R_ijab += -np.einsum('imab,mj->ijab',t2,F_oo)
	R_ijab += - np.einsum('mjab,mi->ijab',t2,F_oo)
	R_ijab += 0.5*np.einsum('mnab,mnij->ijab',t2,W_oooo) 
	R_ijab += 0.5*np.einsum('nmab,nmij->ijab',t2,W_oooo)
	R_ijab += 0.5*np.einsum('ijef,abef->ijab',t2,W_vvvv) 
	R_ijab += 0.5*np.einsum('ijfe,abfe->ijab',t2,W_vvvv)
	R_ijab += np.einsum('imae,mbej->ijab',t2,W_ovvo) 
	R_ijab += - np.einsum('miae,mbej->ijab',t2,W_ovvo)
	R_ijab += np.einsum('imae,mbej->ijab',t2,W_ovvo)
	R_ijab += np.einsum('imae,mbje->ijab',t2,W_ovov)
	R_ijab += np.einsum('mibe,maje->ijab',t2,W_ovov) 
	R_ijab += np.einsum('mjae,mbie->ijab',t2,W_ovov)
	R_ijab += np.einsum('jmbe,maei->ijab',t2,W_ovvo) 
	R_ijab += - np.einsum('mjbe,maei->ijab',t2,W_ovvo)
	R_ijab += np.einsum('jmbe,maei->ijab',t2,W_ovvo) 
	R_ijab += np.einsum('jmbe,maie->ijab',t2,W_ovov)
	
	#      Compute new t2
	t2_new = R_ijab/D
	E_ccd = 2*np.einsum('ijab,ijab',t2_new,twoelecint_mo[:occ,:occ,occ:nao,occ:nao]) - np.einsum('ijab,ijba',t2_new,twoelecint_mo[:occ,:occ,occ:nao,occ:nao])

	#      RMS of t2
	t2_diff = t2_new-t2
	t2_rms = np.sqrt(np.einsum('ijab,ijab',t2_diff,t2_diff))/np.prod(t2.shape)
	del_E = E_ccd - E_old
	if t2_rms <= 1E-10 and abs(del_E) <= 1E-10:
		print "CCD converged!!!"
		print "CCD energy is : "+str(Etot+E_ccd)
		break	
	else:
		print "Cycle number : "+str(k+1)
		print "RMS of t2 : "+str(t2_rms)
		print "Energy difference : "+str(del_E)
		print "energy : "+str(Etot+E_ccd)
		t2 = t2_new
		E_old = E_ccd
