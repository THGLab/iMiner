import sys
sys.path.insert(0, "..")

from iMiner.tools.mutate_model import Mutagenesis

mutagenesis = Mutagenesis("./data/sars_cov_2_helicase.pdb", "./data")
mutagenesis.mutate("A", 179, "GLN")