"""
Python script for performing a fit and immediately calculating test errors after the fit.

Test errors are reported for MAE energy (eV/atom) and MAE force (eV/Angstrom), if using LAMMPS 
metal units.

Serial:

    python example.py

Parallel:

    mpirun -np 2 python example.py

NOTE: See below for info on which variables to change for different options.
"""

from mpi4py import MPI
from fitsnap3lib.fitsnap import FitSnap
from numpy import linalg as LA
import numpy as np

# Declare a communicator (this can be a custom communicator as well).
comm = MPI.COMM_WORLD

# Create an input dictionary containing settings.
settings = \
{
"BISPECTRUM":
    {
    "numTypes": 1,
    "twojmax": 6,
    "rcutfac": 4.67637,
    "rfac0": 0.99363,
    "rmin0": 0.0,
    "wj": 1.0,
    "radelem": 0.5,
    "type": "Ta",
    "wselfallflag": 0,
    "chemflag": 0,
    "bzeroflag": 0,
    "quadraticflag": 0,
    },
"CALCULATOR":
    {
    "calculator": "LAMMPSSNAP",
    "energy": 1,
    "force": 1,
    "stress": 0
    },
"ESHIFT":
    {
    "Ta": 0.0
    },
"SOLVER":
    {
    "solver": "SVD",
    "compute_testerrs": 1,
    "detailed_errors": 1
    },
"SCRAPER":
    {
    "scraper": "JSON" 
    },
"PATH":
    {
    "dataPath": "/Users/adrohsk/FitSNAP/examples/Ta_Linear_JCP2014/JSON"
    },
"OUTFILE":
    {
    "metrics": "Ta_metrics.md",
    "potential": "Ta_pot"
    },
"REFERENCE":
    {
    "units": "metal",
    "atom_style": "atomic",
    "pair_style": "hybrid/overlay zero 10.0 zbl 4.0 4.8",
    "pair_coeff1": "* * zero",
    "pair_coeff2": "* * zbl 73 73"
    },
"EXTRAS":
    {
    "dump_descriptors": 1,
    "dump_truth": 0,
    "dump_weights": 0,
    "dump_dataframe": 0
    },
"GROUPS":
    {
    "group_sections": "name training_size testing_size eweight fweight vweight",
    "group_types": "str float float float float float",
    "smartweights": 0,
    "random_sampling": 0,
    "Displaced_A15" :  "1.0    0.0       1.0 1.0 1.00E-08",
    "Displaced_BCC" :  "1.0    0.0       1.0 1.0               1.00E-08",
    "Displaced_FCC" :  "1.0    0.0       1.0 1.0               1.00E-08",
    "Elastic_BCC"   :  "1.0    0.0       1.0 1.0        0.0001",
    "Elastic_FCC"   :  "1.0    0.0     1.0 1.0        1.00E-09",
    "GSF_110"       :  "1.0    0.0      1.0 1.0               1.00E-08",
    "GSF_112"       :  "1.0    0.0      1.0 1.0               1.00E-08",
    "Liquid"        :  "1.0    0.0      1.0 1.0               1.00E-08",
    "Surface"       :  "1.0    0.0      1.0 1.0               1.00E-08",
    "Volume_A15"    :  "1.0    0.0      1.0 1.0        1.00E-09",
    "Volume_BCC"    :  "1.0    0.0      1.0 1.0        1.00E-09",
    "Volume_FCC"    :  "1.0    0.0      1.0 1.0        1.00E-09"
    },
"MEMORY":
    {
    "override": 0
    }
}

# Alternatively, settings could be provided in a traditional input file:
# settings = "../../Ta_Linear_JCP2014/Ta-example.in"
    
# Create a FitSnap instance using the communicator and settings:
fs = FitSnap(settings, comm=comm, arglist=["--overwrite"])

# Scrape configurations to create and populate the `snap.data` list of dictionaries with structural info.
fs.scrape_configs()
# Calculate descriptors for all structures in the `snap.data` list.
# This is performed in parallel over all processors in `comm`.
# Descriptor data is stored in the shared arrays.
fs.process_configs()
# Now we can access the A matrix of descriptors:
# print(fitsnap.pt.shared_arrays['a'].array)
# Good practice after a large parallel operation is to impose a barrier to wait for all procs to complete.
fs.pt.all_barrier()
# Perform a fit using data in the shared arrays.
fs.perform_fit()
# Can also access the fitsnap dataframe here:
# print(snap.solver.df)
# WriteLAMMPS potential files and error analysis.
fs.write_output()

amat = fs.pt.shared_arrays['a'].array
print(np.shape(amat))
#n = np.shape(amat)[0]
#reg = np.identity(n)*1e-12
#mat = amat + reg
cond = LA.cond(amat)/1e9

print(cond)



