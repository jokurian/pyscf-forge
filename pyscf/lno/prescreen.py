from functools import reduce
import numpy as np
from pyscf import gto, lo, scf,mp,lib
from pyscf.data.elements import chemcore

from dlno import dlno as dlno_mod
from dlno import mp2 as dlno_mp2
from dlno import pao as dlno_pao
from dlno import util as dlno_util
from dlno.domain import _compute_av



def union_objects(obj_list):
    obj_list = [np.asarray(x, dtype=np.int32) for x in obj_list if len(x) > 0]
    if not obj_list:
        return np.empty((0,), dtype=np.int32)
    return reduce(np.union1d, obj_list)


def strong_pair_lists(pair_energy, pair_energy_thr):
    pair_energy_with_self = pair_energy + np.eye(pair_energy.shape[0])
    return [
        np.where(np.abs(pair_energy_with_self[i]) > pair_energy_thr)[0]
        for i in range(pair_energy.shape[0])
    ]


def build_dlno_prescreen_data(
    mf,
    lo_coeff,
    frag_lolist,
    frozen=None,
    lmo_bp_domain_thr=0.999,
    pao_bp_domain_thr=0.98,
    domain_pao_thr=1e-4,
    pair_energy_thr=1e-4,
    multipole_order=4,
):
    """Precompute the DLNO metadata that can later feed a pyscf.lno hookup.

    Returned keys mirror the "Data that should be precomputed once" list from the
    DLNO/LNO theory note:
        - localized occupied LMOs,
        - LMO BP domains and primary domains,
        - PAOs and AO-to-PAO map,
        - domain PAOs and semicanonical local prescreen orbitals,
        - multipole pair-energy matrix,
        - strong-pair lists,
        - fragment-level extended domains and prescreened occupied/virtual orbitals.
    """
    mol = mf.mol
    dlno = dlno_mod.DLNO(mf, frozen=frozen)
    dlno.lmo = lo_coeff
    dlno.lmo_bp_domain_thr = lmo_bp_domain_thr
    dlno.pao_bp_domain_thr = pao_bp_domain_thr
    dlno.domain_pao_thr = domain_pao_thr
    dlno.pair_energy_thr = pair_energy_thr
    dlno.multipole_order = multipole_order

    s1e = dlno.s1e
    fock = dlno.fock
    lmo_bp_domain = dlno.lmo_bp_domain
    lmo_primary_domain = dlno.lmo_primary_domain
    pao = dlno.pao
    ao2pao_map = dlno.ao2pao_map
    pao_bp_domain = dlno.pao_bp_domain

    domain_pao = dlno.build_domain_pao()
    (eo, vo), (ev, vv) = dlno.canonicalize(domain_pao)
    pair_energy = dlno_mp2.pair_energy_multipole(
        mol,
        eo,
        vo,
        ev,
        vv,
        lmo_primary_domain,
        multipole_order,
    )
    strong_pairs = strong_pair_lists(pair_energy, pair_energy_thr)

    extended_bp_domain = [
        union_objects([lmo_bp_domain[j] for j in idx]) for idx in strong_pairs
    ]
    extended_primary_domain = [
        union_objects([lmo_primary_domain[j] for j in idx]) for idx in strong_pairs
    ]
    fragment_data = []
    for ifrag, loidx in enumerate(frag_lolist):
        loidx = np.asarray(loidx, dtype=np.int32)
        frag_strong = union_objects([strong_pairs[i] for i in loidx])
        frag_ext_bp = union_objects([extended_bp_domain[i] for i in loidx])
        frag_ext_primary = union_objects([extended_primary_domain[i] for i in loidx])

        ao_idx = dlno_util.ao_index_by_atom(mol, frag_ext_primary)
        s21 = s1e[ao_idx]
        s22 = s1e[np.ix_(ao_idx, ao_idx)]

        lmo_block = lo_coeff[:, frag_strong]
        lmo_block_prj = dlno_util.project_mo(lmo_block, s21, s22)
        lmo_block_prj = lo.orth.vec_lowdin(lmo_block_prj, s=s22)
        e_occ_prescreen, occ_prescreen = dlno_mod.semicanonicalize(
            mol, lmo_block_prj, fock, frag_ext_primary
        )

        frag_pao = dlno_pao.pao_overlap_with_domain(
            mol,
            pao,
            list(frag_ext_bp),
            ao2pao_map=ao2pao_map,
            s1e=s1e,
            ovlp_thr=domain_pao_thr,
        )
        if frag_pao.shape[1] > 0:
            av = _compute_av(mol, frag_pao, s1e=s1e, atmlst=frag_ext_primary)
            frag_pao = frag_pao[:, av > pao_bp_domain_thr]
        if frag_pao.shape[1] > 0:
            frag_pao_prj = dlno_util.project_mo(frag_pao, s21, s22)
            frag_pao_prj = dlno_util.orthogonalize(occ_prescreen, frag_pao_prj, s22)
            frag_pao_prj = lo.orth.vec_lowdin(frag_pao_prj, s=s22)
            e_vir_prescreen, vir_prescreen = dlno_mod.semicanonicalize(
                mol, frag_pao_prj, fock, frag_ext_primary
            )
        else:
            e_vir_prescreen = np.zeros((0,))
            vir_prescreen = np.zeros((len(ao_idx), 0))

        fragment_data.append(
            {
                "fragment_index": ifrag,
                "lo_indices": loidx,
                "strong_lmo_indices": frag_strong,
                "extended_bp_domain": frag_ext_bp,
                "extended_primary_domain": frag_ext_primary,
                "occ_prescreen_energies": np.asarray(e_occ_prescreen),
                "occ_prescreen_coeff": occ_prescreen,
                "vir_prescreen_energies": np.asarray(e_vir_prescreen),
                "vir_prescreen_coeff": vir_prescreen,
            }
        )

    return {
        "frozen": frozen,
        "lo_coeff": lo_coeff,
        "frag_lolist": frag_lolist,
        "s1e": s1e,
        "fock": fock,
        "lmo_bp_domain": lmo_bp_domain,
        "pao_bp_domain": pao_bp_domain,
        "lmo_primary_domain": lmo_primary_domain,
        "pao": pao,
        "ao2pao_map": ao2pao_map,
        "domain_pao": domain_pao,
        "local_occ_energies": eo,
        "local_occ_orbitals": vo,
        "local_vir_energies": ev,
        "local_vir_orbitals": vv,
        "pair_energy": pair_energy,
        "strong_pairs": strong_pairs,
        "extended_bp_domain": extended_bp_domain,
        "extended_primary_domain": extended_primary_domain,
        "fragment_data": fragment_data,
    }


def print_summary(data):
    pair_energy = data["pair_energy"]
    print()
    print("DLNO precomputed metadata for pyscf.lno prescreening")
    print("====================================================")
    print(f"Number of LMOs                  : {data['lo_coeff'].shape[1]}")
    print(f"Pair-energy matrix shape        : {pair_energy.shape}")
    print(f"Max |pair energy|               : {np.max(np.abs(pair_energy)):.6e}")
    print(f"Mean strong partners per LMO    : {np.mean([len(x) for x in data['strong_pairs']]):.2f}")
    print()

    for frag in data["fragment_data"]:
        print(f"Fragment {frag['fragment_index']}")
        print(f"  LO indices                    : {frag['lo_indices'].tolist()}")
        print(f"  Strong LMOs                   : {frag['strong_lmo_indices'].tolist()}")
        print(f"  Extended BP domain atoms      : {frag['extended_bp_domain'].tolist()}")
        print(f"  Extended primary domain atoms : {frag['extended_primary_domain'].tolist()}")
        print(f"  Occ prescreen size            : {frag['occ_prescreen_coeff'].shape[1]}")
        print(f"  Vir prescreen size            : {frag['vir_prescreen_coeff'].shape[1]}")
        print()


def load_or_run_scf(mf, chkfile, cderi_file=None):
    mf.chkfile = str(chkfile)
    if cderi_file is not None and getattr(mf, "with_df", None) is not None:
        mf.with_df._cderi_to_save = str(cderi_file)
        if cderi_file.exists():
            mf.with_df._cderi = str(cderi_file)
    if chkfile.exists():
        print(f"Loading SCF checkpoint from {chkfile}")
        mf.__dict__.update(lib.chkfile.load(str(chkfile), "scf"))
        mf.converged = True
    else:
        print(f"Running SCF and saving checkpoint to {chkfile}")
        mf.kernel()
    return mf


def load_or_localize_pm(mol, orbocc, lo_coeff_file, localize=None):
    if lo_coeff_file.exists():
        print(f"Loading localized orbitals from {lo_coeff_file}")
        return np.load(lo_coeff_file, allow_pickle=False)

    print(f"Running Pipek-Mezey localization and saving to {lo_coeff_file}")
    if localize is None:
        lo_coeff = lo.PipekMezey(mol, orbocc).kernel()
    else:
        lo_coeff = localize(mol, orbocc)
    np.save(lo_coeff_file, lo_coeff)
    return lo_coeff


def load_or_run_mp2(mf, frozen, mp2_ecorr_file, mp2_factory=None, kernel_kwargs=None):
    if mp2_ecorr_file.exists():
        print(f"Loading MP2 correlation energy from {mp2_ecorr_file}")
        return float(np.load(mp2_ecorr_file, allow_pickle=False))

    print(f"Running MP2 and saving correlation energy to {mp2_ecorr_file}")
    if mp2_factory is None:
        mmp = mp.MP2(mf, frozen=frozen)
    else:
        mmp = mp2_factory(mf, frozen)
    if kernel_kwargs is None:
        kernel_kwargs = {}
    else:
        kernel_kwargs = dict(kernel_kwargs)
    if mp2_factory is None and "with_t2" not in kernel_kwargs:
        kernel_kwargs["with_t2"] = False
    mmp.kernel(**kernel_kwargs)
    e_corr = float(mmp.e_corr)
    np.save(mp2_ecorr_file, np.asarray(e_corr))
    return e_corr
