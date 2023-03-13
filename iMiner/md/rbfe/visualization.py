from typing import Tuple, Union, Optional
import os
from pathlib import Path
from rdkit import Chem
import numpy as np
from pymol import cmd


def perturb_rdmol(molA: Chem.Mol, molB: Chem.Mol, mapping: np.ndarray, pos_offset: Union[float, np.ndarray] = 0.0) -> Tuple[Chem.Mol, np.ndarray]:
    """
    Add dummy atoms for a molecule based on another molecule and mapping info

    Parameters
    ----------
    molA: Chem.Mol
        Molecule to perturb
    molB: Chem.Mol
        Molecule to which molA is perturbed
    mapping: np.ndarray
        Mapping information, with dim Nx2
    pos_offset: np.ndarray or float
        Position offsets added on the conformer. Used in visualization
    
    Returns
    -------
    mol: Chem.Mol
        Perturbed molecule (with dummy atoms added)
    mapping_du: np.ndarray
        Mapping with -1 in molecule A replaced by dummy atom indices 
    """
    duIndices = []
    mapping_du = mapping.copy()
    rwmol = Chem.RWMol(molA)
    newpos = {}
    posA = molA.GetConformer(0).GetPositions()
    posB = molB.GetConformer(0).GetPositions()
    for i in range(mapping.shape[0]):
        atomIdx1 = int(mapping[i, 0])
        atomIdx2 = int(mapping[i, 1])
        if atomIdx1 == -1:
            du = Chem.Atom(0)
            duIdx = rwmol.AddAtom(du)
            mapping_du[i, 0] = duIdx
            duIndices.append((duIdx, atomIdx2))
            newpos[duIdx] = posB[atomIdx2]
        else:
            newpos[atomIdx1] = posA[atomIdx1]

    mapping_B_to_A = {int(m[1]): int(m[0]) for m in mapping_du if m[1] != -1}
    for atomIdx1, atomIdx2 in duIndices:
        for nei in molB.GetAtomWithIdx(atomIdx2).GetNeighbors():
            neiIdx1 = mapping_B_to_A[nei.GetIdx()]
            b = rwmol.GetBondBetweenAtoms(atomIdx1, neiIdx1)
            if not b:
                rwmol.AddBond(atomIdx1, neiIdx1)
    
    mol = rwmol.GetMol()
    mol.RemoveAllConformers()
    conf = Chem.Conformer(mol.GetNumAtoms())
    for idx in newpos:
        conf.SetAtomPosition(idx, newpos[idx] + pos_offset)
    mol.AddConformer(conf, assignId=True)
    return mol, mapping_du


class PyMOLVisualizer:
    def __init__(
        self,
        molA: Union[os.PathLike, Chem.Mol], 
        molB: Union[os.PathLike, Chem.Mol], 
        mapping: Union[os.PathLike, np.ndarray], 
        wdir: os.PathLike = "visualization/",
        molA_name: str = "molA",
        molB_name: str = "molB",
    ):
        self.molA, self.molA_name = self.read_mol(molA)
        self.molB, self.molB_name = self.read_mol(molB)
        if isinstance(mapping, np.ndarray):
            self.mapping = np.array(mapping, dtype=int)
        else:
            self.mapping = np.loadtxt(mapping, dtype=int)
        self.wdir = Path(wdir).resolve()
        self.wdir.mkdir(parents=True, exist_ok=True)
        if (not self.molA_name) or (not self.molB_name) or (self.molA_name == self.molB_name):
            self.molA_name = molA_name
            self.molB_name = molB_name
    
    def read_mol(self, mol: Union[os.PathLike, Chem.Mol]):
        if isinstance(mol, Chem.Mol):
            return mol, mol.GetProp("_Name")
        else:
            suffix = Path(mol).suffix
            name = Path(mol).stem
            if suffix == '.sdf':
                mol = Chem.SDMolSupplier(mol, removeHs=False)[0]
            elif suffix == ".mol":
                mol = Chem.MolFromMolFile(mol, removeHs=False)
            else:
                raise NotImplementedError()
            return mol, name
            
    def perturb(self):
        """
        Perturbation on rdkit mol
        """
        pert_molA, mapping_A_B_pert = perturb_rdmol(self.molA, self.molB, self.mapping)
        ptp = np.ptp(pert_molA.GetConformer().GetPositions(), axis=0)
        pert_molB, mapping_B_A_pert = perturb_rdmol(self.molB, self.molA, self.mapping[:, [1, 0]], ptp)
        self.mapping_du = np.vstack((mapping_A_B_pert[:, 0], mapping_B_A_pert[:, 0])).T

        self.pert_molA_path = str(self.wdir / f'{self.molA_name}.sdf')
        writer = Chem.SDWriter(self.pert_molA_path)
        writer.write(pert_molA)
        writer.close()

        self.pert_molB_path = str(self.wdir / f'{self.molB_name}.sdf')
        writer = Chem.SDWriter(self.pert_molB_path)
        writer.write(pert_molB)
        writer.close()

    def make_pymol_session(self):
        """
        Make PyMOL session to visualize perturbation
        """
        self.perturb()
        session_path = self.wdir / "perturbation.pse"
        if session_path.is_file():
            session_path.unlink()
        cmd.delete("all")
        cmd.bg_color("grey")
        cmd.load(self.pert_molA_path, object=self.molA_name)
        cmd.load(self.pert_molB_path, object=self.molB_name)
        cmd.label(self.molA_name, "rank")
        cmd.label(self.molB_name, "rank")
        perts = []
        for i, j in zip(self.mapping_du[:, 0], self.mapping_du[:, 1]):
            p = f"A{i}-B{j}"
            cmd.select("tmp_a", f"index {i+1} & {self.molA_name}")
            cmd.select("tmp_b", f"index {j+1} & {self.molB_name}")
            cmd.distance(p, 'tmp_a', 'tmp_b', label=0)
            cmd.delete("tmp_a")
            cmd.delete("tmp_b")
            perts.append(p)
        cmd.zoom("visible")
        cmd.save(str(session_path))