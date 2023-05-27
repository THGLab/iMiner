'''
Author: Jie Li
Date created: Oct 3, 2022
This file defines a class for generating molecule library by defining regions and 
fragments in each region.
'''
from rdkit import Chem
from rdkit.Chem import Draw
from copy import deepcopy
from itertools import product

def connect_mols(mols, connector_atom_pairs, bond_order=Chem.rdchem.BondType.SINGLE):
    '''
    Combining multiple molecular fragments into a single molecule

    :param mols: list of rdkit.Chem.rdchem.Mol
    :param connector_atom_pairs: list of [(head_molecule_fragment_idx, head_atom_idx in fragment),
                                    (tail_molecule_fragment_idx, tail_atom_idx in fragment)]
    :param bond_order: Chem.rdchem.BondType
    :return: rdkit.Chem.rdchem.Mol molecule with all fragments connected
    '''
    assert len(mols) > 1, 'At least two fragments are required to connect'
    # mols = [deepcopy(item) for item in mols] # make a copy of the original mols to prevent any unwanted modifications on the original mols
    mols = [Chem.RemoveHs(mol) for mol in mols] # add hydrogens to all fragments
    # for mol in mols:
    #     for atom in mol.GetAtoms():
    #         atom.SetNumExplicitHs(0) # make sure there is no explicit hydrogen atoms
    mol_natoms = [mol.GetNumAtoms() for mol in mols]
    idx_offsets = [sum(mol_natoms[:i]) for i in range(len(mol_natoms))]
    combo = Chem.CombineMols(mols[0], mols[1])
    for i in range(2, len(mols)):
        combo = Chem.CombineMols(combo, mols[i])
    edcombo = Chem.EditableMol(combo)
    for head, tail in connector_atom_pairs:
        edcombo.AddBond(head[1] + idx_offsets[head[0]],
         tail[1] + idx_offsets[tail[0]], bond_order)

    result = edcombo.GetMol()
    Chem.SanitizeMol(result)
    return result

def show_atom_number(mol, label, labelled_idx=None):
    '''https://stackoverflow.com/questions/53321453/rdkit-how-to-show-moleculars-atoms-number'''
    if labelled_idx is None:
        labelled_idx = range(mol.GetNumAtoms())
    mol_with_number = deepcopy(mol)
    for atom in mol_with_number.GetAtoms():
        if atom.GetIdx() in labelled_idx:
            atom.SetProp(label, str(atom.GetIdx()))
    return mol_with_number

class Region():
    def __init__(self, n_connector) -> None:
        self.n_connector = n_connector
        self.size = 0
        self.fragments = []
        self.connect_atom_indices = []

    def add_fragment(self, smiles_or_mol, connector_atoms):
        assert len(connector_atoms) == self.n_connector
        self.size += 1
        if isinstance(smiles_or_mol, str):
            mol = Chem.MolFromSmiles(smiles_or_mol)
        else:
            mol = smiles_or_mol
        self.fragments.append(mol)
        self.connect_atom_indices.append(connector_atoms)

    def show_fragment(self, index):
        labelled_fragment = show_atom_number(self.fragments[index], 'atomNote')
        Draw.MolToImage(labelled_fragment)
        return labelled_fragment

    def show_fragment_gallery(self):
        labelled_frags = [show_atom_number(frag, 'atomNote', labelled_idx) if frag.GetNumAtoms() > 0
                else Chem.MolFromSmiles("*")
                for frag, labelled_idx in zip(self.fragments, self.connect_atom_indices)]
        grid_image = Draw.MolsToGridImage(labelled_frags, legends=[str(i) for i in range(len(labelled_frags))], molsPerRow=4)
        return grid_image

class VirtualLibrary():
    def __init__(self) -> None:
        self.region_count = 0
        self.regions = []
        self.region_connections = []

    def add_region(self, region):
        self.region_count += 1
        self.regions.append(region)
        print("Added region %d with %d fragments" % (self.region_count, region.size))

    def add_region_connection(self, head_region_idx, tail_region_idx):
        self.region_connections.append((head_region_idx-1, tail_region_idx-1))

    def library_size(self):
        size = 1
        for region in self.regions:
            size *= region.size
        return size

    def connect_mol_by_idx(self, indices):
        assert len(indices) == self.region_count
        connector_atom_pairs = []
        for head_region_idx, tail_region_idx in self.region_connections:
            head_connect_atoms = self.regions[head_region_idx].connect_atom_indices[indices[head_region_idx]]
            head_connect_atom_idx = head_connect_atoms[-1]
            tail_connect_atoms = self.regions[tail_region_idx].connect_atom_indices[indices[tail_region_idx]]
            tail_connect_atom_idx = tail_connect_atoms[0]
            connector_atom_pairs.append([(head_region_idx, head_connect_atom_idx), (tail_region_idx, tail_connect_atom_idx)])

        for region_idx, frag_idx in enumerate(indices):
            frag = self.regions[region_idx].fragments[frag_idx]
            if frag.GetNumAtoms() == 0:
                # special handling for empty fragment: not adding this fragment, and connect the previous and next fragments directly together
                modified_connector_atom_pairs = []
                new_pair = []
                for head, tail in connector_atom_pairs:
                    # find which regions the empty fragment are connected to
                    if tail[0] == region_idx:
                        new_pair.append(head)
                    elif head[0] == region_idx:
                        new_pair.append(tail)
                    else:
                        # this connection does not involve the empty fragment
                        modified_connector_atom_pairs.append([head, tail])
                assert len(new_pair) == 2, 'Empty fragment connected to more than two regions!'
                modified_connector_atom_pairs.append(new_pair)
                connector_atom_pairs = modified_connector_atom_pairs


        return connect_mols([self.regions[region_idx].fragments[frag_idx]
                 for region_idx, frag_idx in enumerate(indices)], connector_atom_pairs)

    def generate_library(self):
        all_indices = [list(range(region.size)) for region in self.regions]
        for idx in product(*all_indices):
            yield Chem.MolToSmiles(self.connect_mol_by_idx(idx))


if __name__ == '__main__':
    # library = VirtualLibrary()
    # region1 = Region(1)
    # region1.add_fragment('N1CCCC1', [0])
    # region1.add_fragment("N1C(=O)c2ccccc2C1", [0])
    # region1.add_fragment("N1C(=O)N(C)c2ccccc21", [0])
    # library.add_region(region1)

    # region2 = Region(2)
    # region2.add_fragment("CCCC", [0, 3])
    # region2.add_fragment("CCCC=O", [0, 3])
    # region2.add_fragment("CC(=O)C", [0, 2])
    # region2.add_fragment("", [0, 0])
    # library.add_region(region2)

    # region3 = Region(1)
    # region3.add_fragment("c1ncccc1", [0])
    # region3.add_fragment("c1cc(COC)ccc1", [0])
    # region3.add_fragment("c1cc(CN2CCOCC2)ccc1", [0])
    # library.add_region(region3)

    # print(library.library_size())
    # library.add_region_connection(0, 1)
    # library.add_region_connection(1, 2)
    # library.connect_mol_by_idx([2,2,2])

    fragments_sdf_path = "../../helicase_molecules/tom_virtual_library/"
    library = VirtualLibrary()
    region1 = Region(1)
    for frag in Chem.SDMolSupplier(fragments_sdf_path + "UNC0379_region_1.sdf"):
        region1.add_fragment(frag, [frag.GetIntProp(f"Region {i}") for i in range(1, 5) if frag.GetIntProp(f"Region {i}") != -1])
    library.add_region(region1)

    region2 = Region(2)
    for frag in Chem.SDMolSupplier(fragments_sdf_path + "UNC0379_region_2.sdf"):
        region2.add_fragment(frag, [frag.GetIntProp(f"Region {i}") for i in range(1, 5) if frag.GetIntProp(f"Region {i}") != -1])
    library.add_region(region2)

    region3 = Region(1)
    for frag in Chem.SDMolSupplier(fragments_sdf_path + "UNC0379_region_3.sdf"):
        region3.add_fragment(frag, [frag.GetIntProp(f"Region {i}") for i in range(1, 5) if frag.GetIntProp(f"Region {i}") != -1])
    library.add_region(region3)

    region4 = Region(2)
    for frag in Chem.SDMolSupplier(fragments_sdf_path + "UNC0379_region_4.sdf"):
        region4.add_fragment(frag, [frag.GetIntProp(f"Region {i}") for i in range(1, 5) if frag.GetIntProp(f"Region {i}") != -1])
    library.add_region(region4)

    library.add_region_connection(1 , 2)
    library.add_region_connection(2, 4)
    library.add_region_connection(4, 3)

    i=0
    for item in library.generate_library():
        i+=1
        if i > 10:
            break
        print(item)
