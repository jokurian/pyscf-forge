#Assumes dlno is downloaded from : https://github.com/fishjojo/dlno/ and is available in PYTHONPATH
import time
from pathlib import Path

import numpy as np
from pyscf import gto, lib, lno, lo, mp, scf
from pyscf.lno.prescreen import build_dlno_prescreen_data,load_or_run_scf, load_or_run_mp2, load_or_localize_pm
from pyscf.data.elements import chemcore


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



n_waters = 10
atom = water_chain_atom(n_waters=n_waters, d=8.0)
# atom = "water_21.xyz"
# n_waters = 21
basis = "cc-pvdz"
basis_tag = "".join(ch if ch.isalnum() else "_" for ch in basis)

mol = gto.M(atom=atom, basis=basis, verbose=4, max_memory=8000)
mf = scf.RHF(mol).density_fit()
mf.chkfile = f"scf_{n_waters}.chk"
mf.init_guess = "chk"
# mf.kernel()
df_file = Path(f"df_ints_{n_waters}_{basis_tag}.h5")
mf = load_or_run_scf(mf, Path(mf.chkfile), cderi_file=df_file)
frozen = chemcore(mol)

mmp_e_corr = load_or_run_mp2(
    mf,
    frozen=frozen,
    mp2_ecorr_file=Path(f"mp2_ecorr_{n_waters}_{basis_tag}.npy"),
)


orbocc = mf.mo_coeff[:, frozen : np.count_nonzero(mf.mo_occ)]
lo_coeff = load_or_localize_pm(
    mol,
    orbocc,
    Path(f"lo_coeff_{n_waters}_{basis_tag}.npy"),
)
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

mlcc = lno.LNOCCSD_T(mf, lo_coeff, frag_lolist, frozen=frozen).set(verbose=4)
mlcc.use_dlno_prescreen = True
mlcc.dlno_prescreen_data = dlno_data
mlcc.make_las_variant = "make_las_prescreen"
mlcc.lno_thresh = [1e-5, 1e-6]
mlcc.kernel()

ecc_pt2corrected = mlcc.e_corr_ccsd_pt2corrected(mmp_e_corr)
ecc_t_pt2corrected = mlcc.e_corr_ccsd_t_pt2corrected(mmp_e_corr)

print()
print(f"E_corr(LNO-CCSD)    = {mlcc.e_corr_ccsd: .12f}")
print(f"E_corr(LNO-CCSD(T)) = {mlcc.e_corr_ccsd_t: .12f}")
print()
print(f"E_corr(LNO-CCSD(PT2)) = {ecc_pt2corrected: .12f}")
print(f"E_corr(LNO-CCSD(T)(PT2)) = {ecc_t_pt2corrected: .12f}")
