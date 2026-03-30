import runpy
from pathlib import Path

import numpy as np
from pyscf import gto, lno, lo, scf,lib
from pyscf.lno.prescreen import build_dlno_prescreen_data
from pyscf.data.elements import chemcore
import time

#Assumes dlno is downloaded from : https://github.com/fishjojo/dlno/ and is available in PYTHONPATH
def load_or_run_scf(mf, chkfile):
    mf.chkfile = str(chkfile)
    if chkfile.exists():
        print(f"Loading SCF checkpoint from {chkfile}")
        mf.__dict__.update(lib.chkfile.load(str(chkfile), "scf"))
    else:
        print(f"Running SCF and saving checkpoint to {chkfile}")
        mf.kernel()
    return mf

def water_chain_atom(n_waters, d):
    r_oh = 0.9572
    angle = np.deg2rad(104.52)

    o = np.array([0.0, 0.0, 0.0])
    h1 = np.array([r_oh * np.sin(angle / 2), 0.0, r_oh * np.cos(angle / 2)])
    h2 = np.array([-r_oh * np.sin(angle / 2), 0.0, r_oh * np.cos(angle / 2)])

    lines = []
    for i in range(n_waters):
        shift = np.array([i * d, 0.0, 0.0])
        lines.append(f"O {*(o + shift),}")
        lines.append(f"H {*(h1 + shift),}")
        lines.append(f"H {*(h2 + shift),}")
    return "\n".join(lines)



n_waters = 20
atom = water_chain_atom(n_waters=n_waters, d=8.0)
# atom = "water_21.xyz"
# n_waters = 21
basis = "cc-pvdz"

mol = gto.M(atom=atom, basis=basis, verbose=4, max_memory=8000)
mf = scf.RHF(mol).density_fit()
mf.chkfile = f"scf_{n_waters}.chk"
mf.init_guess = "chk"
# mf.kernel()
mf = load_or_run_scf(mf, Path(mf.chkfile))

frozen = chemcore(mol)
orbocc = mf.mo_coeff[:, frozen : np.count_nonzero(mf.mo_occ)]
lo_coeff = lo.PipekMezey(mol, orbocc).kernel()
frag_lolist = [[i] for i in range(lo_coeff.shape[1])]

time0 = time.time()
dlno_data = build_dlno_prescreen_data(
    mf,
    lo_coeff,
    frag_lolist,
    frozen=frozen,
    lmo_bp_domain_thr=0.9999,
    pao_bp_domain_thr=0.98,
    domain_pao_thr=1e-5,
    pair_energy_thr=1e-5,
    multipole_order=4,
)
time1 = time.time()
print(f"DLNO prescreening data built in {time1 - time0: .2f} seconds")
from pympler import asizeof
deep_size = asizeof.asizeof(dlno_data)
print(f"Deep size of dictionary: {deep_size / (1024**2): .2f} mb")

mlcc = lno.LNOCCSD_T(mf, lo_coeff, frag_lolist, frozen=frozen).set(verbose=4)
mlcc.use_dlno_prescreen = True
mlcc.dlno_prescreen_data = dlno_data
#mlcc.make_las_variant = make_las_prescreen #can manually change the function: eg. from pyscf.lno.lno import make_las_prescreen. By default uses mlcc.use_dlno_prescreen to decide which to take
mlcc.lno_thresh = [1e-5, 1e-6]
mlcc.kernel()

print()
print(f"E_corr(LNO-CCSD)    = {mlcc.e_corr_ccsd: .12f}")
print(f"E_corr(LNO-CCSD(T)) = {mlcc.e_corr_ccsd_t: .12f}")
print()
