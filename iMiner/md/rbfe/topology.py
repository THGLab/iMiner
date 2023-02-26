"""
Author: Eric Wang
Date: 02/26/2023

Class to manipulate topology with perturbation in Gromacs RBFE calculation
"""
import warnings
from typing import Dict, Tuple, Optional
from itertools import permutations
import os

import numpy as np
import parmed


class GromacsTopologyFilePerturb(parmed.gromacs.GromacsTopologyFile):
    """
    A chemical structure composed of atoms, bonds, angles, torsions, and other topological features.
    Inherited from `parmed.gromacs.GromacsTopologyFile` and add features to support perturbed topology
    """
    @property
    def bonds_dict(self) -> Dict[Tuple[int, int], parmed.Bond]:
        if hasattr(self, "_bonds_dict"):
            return self._bonds_dict
        else:
            self._bonds_dict = {}
            for bond in self.bonds:
                self._bonds_dict[(bond.atom1.idx, bond.atom2.idx)] = bond
            return self._bonds_dict
    
    def get_bond_with_atom_idx(self, atom1: int, atom2: int) -> parmed.Bond:
        try:
            bond = self.bonds_dict[(atom1, atom2)]
        except KeyError as e:
            bond = self.bonds_dict[(atom2, atom1)]
        return bond
    
    @property
    def adjusts_dict(self) -> Dict[Tuple[int, int], parmed.NonbondedException]:
        if hasattr(self, "_adjusts_dict"):
            return self._adjusts_dict
        else:
            self._adjusts_dict = {}
            for adjust in self.adjusts:
                self._adjusts_dict[(adjust.atom1.idx, adjust.atom2.idx)] = adjust
            return self._adjusts_dict
    
    def get_adjust_with_atom_idx(self, atom1: int, atom2: int) -> parmed.NonbondedException:
        try:
            adjust = self.adjusts_dict[(atom1, atom2)]
        except KeyError as e:
            adjust = self.adjusts_dict[(atom2, atom1)]
        return adjust
    
    @property
    def angles_dict(self) -> Dict[Tuple[int, int, int], parmed.Angle]:
        if hasattr(self, "_angles_dict"):
            return self._angles_dict
        else:
            self._angles_dict = {}
            for ang in self.angles:
                self._angles_dict[(ang.atom1.idx, ang.atom2.idx, ang.atom3.idx)] = ang
            return self._angles_dict
    
    def get_angle_with_atom_idx(self, atom1: int, atom2: int, atom3: int) -> parmed.Angle:
        try:
            ang = self.angles_dict[(atom1, atom2, atom3)]
        except KeyError as e:
            ang = self.angles_dict[(atom3, atom2, atom1)]
        return ang
    
    @property
    def proper_dihedrals_dict(self) -> Dict[Tuple[int, int, int, int], parmed.Dihedral]:
        if hasattr(self, "_proper_dihedrals_dict"):
            return self._proper_dihedrals_dict
        else:
            self._proper_dihedrals_dict = {}
            for dihe in self.dihedrals:
                if not dihe.improper:
                    self._proper_dihedrals_dict[(dihe.atom1.idx, dihe.atom2.idx, dihe.atom3.idx, dihe.atom4.idx)] = dihe
            return self._proper_dihedrals_dict
    
    def get_proper_dihedral_with_atom_idx(self, atom1: int, atom2: int, atom3: int, atom4: int) -> parmed.Dihedral:
        try:
            dihe = self.proper_dihedrals_dict[(atom1, atom2, atom3, atom4)]
        except KeyError as e:
            dihe = self.proper_dihedrals_dict[(atom4, atom3, atom2, atom1)]
        return dihe

    @property
    def improper_dihedrals_dict(self) -> Dict[Tuple[int, int, int, int], parmed.Dihedral]:
        if hasattr(self, "_improper_dihedrals_dict"):
            return self._improper_dihedrals_dict
        else:
            self._improper_dihedrals_dict = {}
            for dihe in self.dihedrals:
                if dihe.improper:
                    self._improper_dihedrals_dict[(dihe.atom1.idx, dihe.atom2.idx, dihe.atom3.idx, dihe.atom4.idx)] = dihe
            return self._improper_dihedrals_dict
    
    def get_improper_dihedral_with_atom_idx(self, atom1: int, atom2: int, atom3: int, atom4: int) -> parmed.Dihedral:
        dihe = None
        for ls in permutations((atom1, atom2, atom3, atom4)):
            key = (ls[0], ls[1], ls[2], ls[3])
            if key in self._improper_dihedrals_dict:
                dihe = self._improper_dihedrals_dict[key]
                break
        if dihe:
            return dihe
        else:
            raise KeyError("No dihedrals found with " + '-'.join(f'Atom{i}' for i in ls))
    
    def add_dummy_atom_type(self, du_type='du'):
        """
        Add dummy atom types to `parmed.Structure.parameterset.atom_types`
        """
        atypes = self.parameterset.atom_types
        if du_type not in atypes:
            du_atype = parmed.AtomType(name=du_type, number=None, mass=0, atomic_number=0, bond_type=du_type, charge=0.0)
            du_atype.sigma = 0
            du_atype.epsilon = 0
            atypes[du_type] = du_atype

    def add_dummy_atom(self, du_type: str = 'du', du_name: str = 'DU', mass: float = 0.0) -> parmed.Atom:
        """
        Add dummy atom
        """
        self.add_dummy_atom_type(du_type)
        n_du = sum([at.type == du_type for at in self.atoms])
        du_atom = parmed.Atom(atomic_number=0, name=f'{du_name}{n_du+1}', type=du_type, charge=0.0, mass=mass)
        self.add_atom(
            du_atom, self.residues[-1].name, self.residues[-1].number
        )
        return du_atom
    
    def get_atom_with_idx(self, idx: int) -> parmed.Atom:
        res = None
        for at in self.atoms:
            if at.idx == idx:
                res = at
                break
        assert res, f"No atom with index {idx} is found"
        return res
    
    def perturb(self, other, mapping):
        
        if not isinstance(other, self.__class__):
            raise TypeError("Not `iMiner.md.rbfe.topolgy.Structure` instance")
        
        # add dummy types
        self.add_dummy_atom_type("du")
        self.add_dummy_atom_type("du_h")

        mapping = np.array(mapping, dtype=int)
        mapping_list = []
        for i in range(mapping.shape[0]):
            if mapping[i, 1] >= 0:
                atomB = other.get_atom_with_idx(mapping[i, 1])
                if mapping[i, 0] == -1:
                    if atomB.element_name == "H":
                        atomA = self.add_dummy_atom("du_h", "DH", atomB.mass)
                    else:
                        atomA = self.add_dummy_atom("du", "DU", atomB.mass)              
                else:
                    atomA = self.get_atom_with_idx(mapping[i, 0])
                # set atom type B
                atomA.typeB = atomB.type
                atomA.chargeB = atomB.charge
                atomA.massB = atomB.mass  
                mapping_list.append((atomA, atomB))
            else:
                atomA = self.get_atom_with_idx(mapping[i, 0])
                atomA.typeB = "du" if atomA.element_name != "H" else "du_h"
                atomA.chargeB = 0.0
                atomA.massB = atomA.mass
        
        map_A_to_B = {aA: aB for aA, aB in mapping_list}
        map_B_to_A = {aB: aA for aA, aB in mapping_list}
        
        # Bonds
        for bondB in other.bonds:
            atom1 = map_B_to_A[bondB.atom1]
            atom2 = map_B_to_A[bondB.atom2]
            try:
                self.get_bond_with_atom_idx(atom1.idx, atom2.idx)
            except KeyError as e:
                # copy bonds from other to self
                btype = parmed.BondType(bondB.type.k, bondB.type.req, self.bond_types)
                bond = parmed.Bond(atom1, atom2, btype, bondB.order)
                self.bonds.append(bond)
        
        for bond in self.bonds:
            try:
                atom1, atom2 = map_A_to_B[bond.atom1], map_A_to_B[bond.atom2]
                bondB = other.get_bond_with_atom_idx(atom1.idx, atom2.idx)
                bond.typeB = parmed.BondType(bondB.type.k, bondB.type.req)
            except KeyError as e:
                bond.typeB = parmed.BondType(bond.type.k, bond.type.req)
        
        # 1-4 Pairs
        for adjustB in other.adjusts:
            atom1 = map_B_to_A[adjustB.atom1]
            atom2 = map_B_to_A[adjustB.atom2]
            try:
                self.get_adjust_with_atom_idx(atom1.idx, atom2.idx)
            except KeyError as e:
                # copy adjusts from other to self
                nbetype = parmed.NonbondedExceptionType(
                    adjustB.type.rmin, adjustB.type.epsilon, adjustB.type.chgscale, self.adjust_types
                )
                nbe = parmed.NonbondedException(atom1, atom2, nbetype)
                self.adjusts.append(nbe)
        
        # Angles
        for angleB in other.angles:
            atom1 = map_B_to_A[angleB.atom1]
            atom2 = map_B_to_A[angleB.atom2]
            atom3 = map_B_to_A[angleB.atom3]
            try:
                self.get_angle_with_atom_idx(atom1.idx, atom2.idx, atom3.idx)
            except KeyError as e:
                # copy angles from other to self
                angtype = parmed.AngleType(angleB.type.k, angleB.type.theteq, self.angle_types)
                angle = parmed.Angle(atom1, atom2, atom3, angtype)
                self.angles.append(angle)
        
        for angle in self.angles:
            try:
                atom1, atom2, atom3 = map_A_to_B[angle.atom1], map_A_to_B[angle.atom2], map_A_to_B[angle.atom3]
                angleB = other.get_angle_with_atom_idx(atom1.idx, atom2.idx, atom3.idx)
                angle.typeB = parmed.AngleType(angleB.type.k, angleB.type.theteq)
            except KeyError as e:
                angle.typeB = parmed.AngleType(angle.type.k, angle.type.theteq)
        
        # Dihedrals
        for diheB in other.dihedrals:
            atom1 = map_B_to_A[diheB.atom1]
            atom2 = map_B_to_A[diheB.atom2]
            atom3 = map_B_to_A[diheB.atom3]
            atom4 = map_B_to_A[diheB.atom4]
            try:
                if diheB.improper:
                    self.get_improper_dihedral_with_atom_idx(atom1.idx, atom2.idx, atom3.idx, atom4.idx)
                else:
                    self.get_proper_dihedral_with_atom_idx(atom1.idx, atom2.idx, atom3.idx, atom4.idx)
            except KeyError as e:
                # copy dihedrals from other to self
                if diheB.improper:
                    dtype = parmed.DihedralType(
                        diheB.type.phi_k, diheB.type.per, diheB.type.phase, 
                        diheB.type.scee, diheB.type.scnb, self.dihedral_types
                    )
                else:
                    dtype = parmed.DihedralTypeList(list=self.dihedral_types)
                    for dtypeItemB in diheB.type:
                        dtypeItem = parmed.DihedralType(
                            dtypeItemB.phi_k, dtypeItemB.per, dtypeItemB.phase,
                            dtypeItemB.scee, dtypeItemB.scnb, None
                        )
                        dtype.append(dtypeItem)
                dihe = parmed.Dihedral(
                    atom1, atom2, atom3, atom4, 
                    improper=diheB.improper, ignore_end=diheB.ignore_end, type=dtype
                )
                self.dihedrals.append(dihe)
        
        for dihe in self.dihedrals:
            try:
                atom1 = map_A_to_B[dihe.atom1]
                atom2 = map_A_to_B[dihe.atom2]
                atom3 = map_A_to_B[dihe.atom3]
                atom4 = map_A_to_B[dihe.atom4]
                if dihe.improper:
                    diheB = other.get_improper_dihedral_with_atom_idx(atom1.idx, atom2.idx, atom3.idx, atom4.idx)
                    dihe.typeB = parmed.DihedralType(
                        diheB.type.phi_k, diheB.type.per, diheB.type.phase,
                        diheB.type.scee, diheB.type.scnb, None
                    )
                else:
                    diheB = other.get_proper_dihedral_with_atom_idx(atom1.idx, atom2.idx, atom3.idx, atom4.idx)
                    dtype = parmed.DihedralTypeList()
                    for dtypeItemB in diheB.type:
                        dtypeItem = parmed.DihedralType(
                            dtypeItemB.phi_k, dtypeItemB.per, dtypeItemB.phase,
                            dtypeItemB.scee, dtypeItemB.scnb, None
                        )
                        dtype.append(dtypeItem)
                        # If one peroidicity is in B but is not in A, add it in A
                        if dtypeItemB.per not in [d.per for d in dihe.type]:
                            dihe.type.append(parmed.DihedralType(
                                0.0, dtypeItemB.per, dtypeItemB.phase, dtypeItemB.scee, dtypeItemB.scnb
                            ))
                    # If one peroidicity is in A but not in B, add it in B
                    for dtypeItem in dihe.type:
                        if dtypeItem.per not in [d.per for d in dtype]:
                            dtype.append(parmed.DihedralType(
                                0.0, dtypeItem.per, dtypeItem.phase, dtypeItem.scee, dtypeItem.scnb
                            ))
                    dihe.typeB = dtype
                    # sort the dihedral types in peroidicty
                    dihe.type.sort(key=lambda d: d.per)
                    dihe.typeB.sort(key=lambda d: d.per)
            except KeyError as e:
                # this term does not exist in B, just copy the term in A as B
                if dihe.improper:
                    dihe.typeB = parmed.DihedralType(
                        dihe.type.phi_k, dihe.type.per, dihe.type.phase,
                        dihe.type.scee, dihe.type.scnb, None
                    )
                else:
                    dihe.typeB = parmed.DihedralTypeList()
                    for dtypeItem in dihe.type:
                        dtypeItemB = parmed.DihedralType(
                            dtypeItem.phi_k, dtypeItem.per, dtypeItem.phase,
                            dtypeItem.scee, dtypeItem.scnb, None
                        )
                        dihe.typeB.append(dtypeItemB)

    def write(self, dest: Optional[os.PathLike] = None, itp: bool = False) -> str:
        """
        Constuct gromacs topology file with perturbations, based on Structure of both states

        This function is rewritten from https://github.com/ParmEd/ParmEd/blob/master/parmed/gromacs/gromacstop.py
        """
        out = []
        if self.defaults.gen_pairs == "no":
            raise NotImplementedError("Topology with gen-pairs = 'no' is not supported")
    
        if not itp:
            out.append('\n[ defaults ]\n')
            out.append('; nbfunc        comb-rule       gen-pairs       fudgeLJ fudgeQQ\n')
            out.append(
                f'{self.defaults.nbfunc:<15d} {self.defaults.comb_rule:<15d} {self.defaults.gen_pairs:<15s} '
                f'{self.defaults.fudgeLJ:<12.8g} {self.defaults.fudgeQQ:<12.8g}\n\n'
            )
        
        # atom types
        econv = parmed.unit.kilocalories.conversion_factor_to(parmed.unit.kilojoules)
        out.append("[ atomtypes ]\n")
        out.append('; name    ')

        # ensure parameterset is not None
        if self.parameterset is None:
            pset = parmed.ParameterSet.from_structure(self)
            self.parameterset = pset
        
        atomTypes = self.parameterset.atom_types
        print_bond_types = all([at._bond_type is not None for at in atomTypes.values()])
        print_atnum = all([at.atomic_number != -1 for at in atomTypes.values()])
        if print_bond_types: out.append('bond_type ')
        if print_atnum: out.append('at.num    ')
        out.append('mass    charge ptype  sigma      epsilon\n')
        for key in atomTypes.keys():
            at = atomTypes[key]
            out.append(f'{str(at):<7s} ')
            if print_bond_types:
                out.append(f'{str(at.bond_type):<8s} ')
            if print_atnum:
                out.append('%8d ' % at.atomic_number)
            out.append(
                f'{at.mass:10.6f}  {at.charge:10.8f}  A '
                f'{at.sigma/10:14.8f} {at.epsilon*econv:14.8f}\n'
            )
        out.append('\n')

        # moleculetype
        self.title = "system" if not self.title.strip() else self.title  # title must not be whitespace only
        out.append('\n[ moleculetype ]\n; Name            nrexcl\n')
        out.append(f'{self.title}          {self.nrexcl}\n\n')

        # atoms
        out.append('[ atoms ]\n')
        out.append(';   nr       type  resnr residue  atom   cgnr    charge       mass  typeB    chargeB      massB\n')
        runchgA, runchgB = 0, 0
        for res in self.residues:
            out.append(
                f'; residue {res.idx+1:4d} {res.name} rtp {res.name} q {sum(a.charge for a in res):.1f} \n'
            )
            for atom in res.atoms:
                runchgA += atom.charge
                runchgB += atom.chargeB
                out.append(
                    f'{atom.idx+1:5d} {atom.type:10s} {res.idx+1:6d} {res.name:6s} {atom.name:6s} {atom.idx+1:6d} '
                    f'{atom.charge:10.8f} {atom.mass:10.6f} {atom.typeB:10s} {atom.chargeB:10.8f} {atom.massB:10.6f} '
                    f';  qtotA {runchgA:10.6f}  qtotB {runchgB:10.6f}\n'
                )
        out.append('\n')

        # extra points
        EPs = [a for a in self.atoms if isinstance(a, parmed.ExtraPoint)]
        settle = False
        if len(self.atoms) - len(EPs) == 3:
            try:
                oxy, = (a for a in self.atoms if a.atomic_number == 8)
                hyd1, hyd2 = (a for a in self.atoms if a.atomic_number == 1)
                settle = True
            except ValueError:
                pass

        # bonds
        if self.bonds:
            conv = (parmed.unit.kilocalorie_per_mole / parmed.unit.angstrom ** 2).conversion_factor_to(
                parmed.unit.kilojoule_per_mole / parmed.unit.nanometer ** 2) * 2
            if settle:
                out.append('#ifdef FLEXIBLE\n\n')
            out.append('[ bonds ]\n')
            out.append(f"; {'ai':6s} {'aj':6s} {'funct':5s} {'b0':10s} {'kb':10s} {'b0B':10s} {'kbB':10s}\n")
            for bond in self.bonds:
                if isinstance(bond.atom1, parmed.ExtraPoint) or isinstance(bond.atom2, parmed.ExtraPoint):
                    continue # pragma: no cover
                out.append(
                    f'{bond.atom1.idx + 1:7d} {bond.atom2.idx + 1:6d} {bond.funct:5d} '
                    f'{bond.type.req / 10:.5f} {bond.type.k * conv:.2f} '
                    f'{bond.typeB.req / 10:.5f} {bond.typeB.k * conv:.2f} '
                    '\n'
                )
            out.append('\n')
        
        # Do the pair-exceptions, currently we only support gen-pairs == yes in defaults
        if self.adjusts:
            out.append('[ pairs ]\n')
            out.append(f"; {'ai':6s} {'aj':6s} {'funct':5s}\n")
            for ad in self.adjusts:
                out.append(f'{ad.atom1.idx+1:7d} {ad.atom2.idx+1:6d} {ad.funct:5d}\n')
            out.append('\n')
        else:
            warnings.warn("The topology has no 1-4 pairs")

        # Angles
        if self.angles:
            conv = (parmed.unit.kilocalorie_per_mole/parmed.unit.radian ** 2).conversion_factor_to(
                parmed.unit.kilojoule_per_mole/parmed.unit.radian ** 2 ) * 2
            out.append('[ angles ]\n')
            out.append(f"; {'ai':6s} {'aj':6s} {'ak':6s} {'funct':5s} {'th0':10s} {'cth':10s} {'th0B':10s} {'cthB':10s}\n")
            # sort
            for angle in self.angles:
                out.append(
                    f"{angle.atom1.idx + 1:7d} {angle.atom2.idx + 1:6d} {angle.atom3.idx + 1:6d} {angle.funct:5d} "
                    f"{angle.type.theteq:.7f} {angle.type.k * conv:.7f} {angle.typeB.theteq:.7f} {angle.typeB.k * conv:.7f} "
                    "\n"
                )
                
                if angle.funct == 5:
                    # Find the Urey-Bradley term, if it exists
                    # TODO: support Urey-Bradley term
                    raise NotImplementedError("Urey-Bradley term is not supported yet")
            out.append('\n')
        
        # Dihedrals
        if self.dihedrals:
            out.append('[ dihedrals ]\n')
            out.append(
                f"; {'ai':6s} {'aj':6s} {'ak':6s} {'funct':5s} "
                f" {'phi0':10s} {'cp':10s} {'mult':10s} "
                f" {'phi0B':10s} {'cpB':10s} {'multB':10s}\n"
            )
            conv = parmed.unit.kilocalories.conversion_factor_to(parmed.unit.kilojoules)
            for dihe in self.dihedrals:
                if dihe.improper:
                    dtypesA, dtypesB = [dihe.type], [dihe.typeB]
                else:
                    dtypesA, dtypesB = dihe.type, dihe.typeB
                for dtA, dtB in zip(dtypesA, dtypesB):
                    out.append(
                        f"{dihe.atom1.idx + 1:7d} {dihe.atom2.idx + 1:6d} {dihe.atom3.idx + 1:6d} {dihe.atom4.idx + 1:6d} {dihe.funct:5d} "
                        f"{dtA.phase:.5f} {dtA.phi_k * conv:.7f} {int(dtA.per):d} "
                        f"{dtB.phase:.5f} {dtB.phi_k * conv:.7f} {int(dtB.per):d} "
                        "\n"
                    )
            out.append('\n')
        
        # RB-torsions
        if self.rb_torsions:
            raise NotImplementedError("RB-torsions are not supported yet")

        # Impropers
        if self.impropers:
            raise NotImplementedError("Impropers are not supported yet")

        # Cmaps
        if self.cmaps:
            raise NotImplementedError("Cmaps are not supported yet")

        if settle:
            raise NotImplementedError("[ settle ] are not supported yet")
                
        if EPs:
            raise NotImplementedError("virtual sites are not supported yet")
    
        # Do we need to list exclusions for systems with EPs?
        if EPs or settle:
            raise NotImplementedError("[ exclusions ] are not supported yet")
        
        if not itp:
            out.append(f"\n[ system ]\n; Name\n{self.title}\n")
            out.append(f"\n[ molecules ]\n; Compound   nmols\n{self.title}  1\n")
        outstr = "".join(out)
        if dest:
            with open(dest, 'w') as f:
                f.write(outstr)
        return outstr

