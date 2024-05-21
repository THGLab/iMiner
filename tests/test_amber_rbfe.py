from pathlib import Path
from iMiner.md.amber.rbfe import AmberRbfeProject


def test_amber_rbfe():
    data_dir = Path(__file__).parent / 'data/rbfe'
    wdir = Path(__file__).parent / 'test_rbfe_project'
    proj = AmberRbfeProject(wdir=wdir)
    proj.add_ligand(data_dir / 'CDD_1845.sdf', name='CDD_1845', charge_method='gas')
    proj.add_ligand(data_dir / 'CDD_1819.sdf', name='CDD_1819', charge_method='gas')
    proj.add_protein(data_dir / 'CDD_1845.pdb', name='CDD_1845')
    proj.add_perturbation('CDD_1845', 'CDD_1819', 'CDD_1845', mcs=data_dir/'mcs.sdf', config=data_dir/'config.json')
