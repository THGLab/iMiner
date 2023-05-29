from generate_fragment_library import *

library = VirtualLibrary()
region1 = Region(1)
region1.add_fragment('C1C=CC2C=CC=CC=2C=1', [5])
region1.add_fragment("C1C=CC2N=CC=CC=2C=1", [5])
region1.add_fragment("C1C=CC2N=CN=CC=2C=1", [5])
region1.add_fragment("C1C=CC2C=CN=CC=2C=1", [5])

library.add_region(region1)   # asdf

region2 = Region(2)
region2.add_fragment("C(NC1C=CC=CC=1)=O", [0, 3])
library.add_region(region2)


region3 = Region(1)
region3.add_fragment("C1C2C(=O)N(C)C=NC=2C=CC=1", [6])
region3.add_fragment("C1C2N=CNC=2C=CC=1", [3])
region3.add_fragment("C1C2N=COC=2C=CC=1", [3])
region3.add_fragment("C1C2C(=O)NCC=2C=CC=1", [4])
region3.add_fragment("C1C2N=CC=NC=2C=CC=1", [4])
library.add_region(region3)

library.add_region_connection(1, 2)
library.add_region_connection(2, 3)

for item in library.generate_library():
    print(item)