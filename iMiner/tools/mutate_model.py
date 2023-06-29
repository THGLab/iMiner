"""
The majority of the code is from the Modeller tutorial:
https://salilab.org/modeller/wiki/Mutate_model
with some modifications to make it work with iMiner.

The original code is licensed under the following license:
https://salilab.org/modeller/registration.html

The code is modified by iMiner developers and is licensed under the MIT license.
Date of modification: 06/26/2023
"""

import os
from pathlib import Path

from modeller import *
from modeller.optimizers import MolecularDynamics, ConjugateGradients
from modeller.automodel import autosched

def optimize(atmsel, sched):
    #conjugate gradient
    for step in sched:
        step.optimize(atmsel, max_iterations=200, min_atom_shift=0.001)
    #md
    refine(atmsel)
    cg = ConjugateGradients()
    cg.optimize(atmsel, max_iterations=200, min_atom_shift=0.001)


#molecular dynamics
def refine(atmsel):
    # at T=1000, max_atom_shift for 4fs is cca 0.15 A.
    md = MolecularDynamics(cap_atom_shift=0.39, md_time_step=4.0,
                           md_return='FINAL')
    init_vel = True
    for (its, equil, temps) in ((200, 20, (150.0, 250.0, 400.0, 700.0, 1000.0)),
                                (200, 600,
                                 (1000.0, 800.0, 600.0, 500.0, 400.0, 300.0))):
        for temp in temps:
            md.optimize(atmsel, init_velocities=init_vel, temperature=temp,
                         max_iterations=its, equilibrate=equil)
            init_vel = False


#use homologs and dihedral library for dihedral angle restraints
def make_restraints(mdl1, aln):
   rsr = mdl1.restraints
   rsr.clear()
   s = Selection(mdl1)
   for typ in ('stereo', 'phi-psi_binormal'):
       rsr.make(s, restraint_type=typ, aln=aln, spline_on_site=True)
   for typ in ('omega', 'chi1', 'chi2', 'chi3', 'chi4'):
       rsr.make(s, restraint_type=typ+'_dihedral', spline_range=4.0,
                spline_dx=0.3, spline_min_points = 5, aln=aln,
                spline_on_site=True)

triplet_to_singlet = {
    'ALA': 'A',
    'ARG': 'R',
    'ASN': 'N',
    'ASP': 'D',
    'CYS': 'C',
    'GLN': 'Q',
    'GLU': 'E',
    'GLY': 'G',
    'HIS': 'H',
    'ILE': 'I',
    'LEU': 'L',
    'LYS': 'K',
    'MET': 'M',
    'PHE': 'F',
    'PRO': 'P',
    'SER': 'S',
    'THR': 'T',
    'TRP': 'W',
    'TYR': 'Y',
    'VAL': 'V'
}

class Mutagenesis:
    """
    mutagenesis class that takes in a pdb file and a residue position and
    mutates the residue to the desired type    
    """
    def __init__(self, pdb_file, temp_path):
        self.pdb_file = pdb_file
        self.protein_name = Path(pdb_file).stem
        self.temp_path = Path(temp_path)
        self.temp_path.mkdir(parents=True, exist_ok=True)
        

    def mutate(self, chain, respos, resname, radius = 10):
        """
        mutates the residue in the pdb file and saves the mutated pdb file

        Args:
            mutation_list (list): list of mutations to be made in the order of chain, residue position, 
            residue name to be mutated.
        """
        env = Environ()
        env.io.hetatm = True
        #soft sphere potential
        env.edat.dynamic_sphere=False
        #lennard-jones potential (more accurate)
        env.edat.dynamic_lennard=True
        env.edat.contact_shell = 4.0
        env.edat.update_dynamic = 0.39

        env.io.atom_files_directory = ['.', '../atom_files']
        env.libs.topology.read(file='$(LIB)/top_heav.lib')
        env.libs.parameters.read(file='$(LIB)/par.lib')

        mdl1 = Model(env, file=self.pdb_file)
        aln = Alignment(env)
        aln.append_model(mdl1, atom_files=self.protein_name, align_codes=self.protein_name)

        orginal_aa = triplet_to_singlet[mdl1.chains[chain].residues[str(respos)].name]
        print("*"*50)
        print("original aa: ", orginal_aa)
        mutated_aa = triplet_to_singlet[resname]
        
        # mutate residue types
        s = Selection(mdl1.chains[chain].residues[str(respos)])
        s.mutate(residue_type=resname)
        
        mutation_info = '_' + chain + '_' + orginal_aa + str(respos) + mutated_aa
        aln.append_model(mdl1, align_codes=self.protein_name)

        mdl1.clear_topology()
        mdl1.generate_topology(aln[-1])
        mdl1.transfer_xyz(aln)

        # Build the remaining unknown coordinates
        mdl1.build(initialize_xyz=False, build_method='INTERNAL_COORDINATES')
        mdl2 = Model(env, file=self.pdb_file)
        mdl1.res_num_from(mdl2,aln)

        #It is usually necessary to write the mutated sequence out and read it in
        #before proceeding, because not all sequence related information about MODEL
        #is changed by this command (e.g., internal coordinates, charges, and atom
        #types and radii are not updated).
        mdl1.write(file=self.protein_name+mutation_info+'.tmp')
        mdl1.read(file=self.protein_name+mutation_info+'.tmp')


        make_restraints(mdl1, aln)
        mdl1.env.edat.nonbonded_sel_atoms=1
        sched = autosched.loop.make_for_model(mdl1)

        #only optimize the selected residue (in first pass, just atoms in selected
        #residue, in second pass, include nonbonded neighboring atoms)
        #set up the mutate residue selection segment
        s = Selection(mdl1.chains[chain].residues[str(respos)].atoms['CA'].select_sphere(radius))

        mdl1.restraints.unpick_all()
        mdl1.restraints.pick(s)

        s.energy()
        s.randomize_xyz(deviation=4.0)

        mdl1.env.edat.nonbonded_sel_atoms=2
        optimize(s, sched)
        mdl1.env.edat.nonbonded_sel_atoms=1
        optimize(s, sched)

        s.energy()
        output_fp = self.temp_path / (self.protein_name + mutation_info + '.pdb')
        mdl1.write(file=str(output_fp))

        #delete the temporary file
        os.remove(self.protein_name+mutation_info+'.tmp')
        return True
    
if __name__ == '__main__':

    mutagenesis = Mutagenesis("../../examples/data/sars_cov_2_helicase.pdb", "./data")
    mutagenesis.mutate("A", 179, "GLN")