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
"ACE":
    {
    "numTypes": 2,
    "rcutfac": "5.0  5.0 5.0 5.0",
    "lambda": "1.5 1.5 1.5 1.5",
    "rcinner": "1.1 1.1 1.1 1.1",
    "drcinner": "0.01 0.01 0.01 0.01",
    "ranks": "1 2 3",
    "lmax":  "1 2 3",
    "nmax": "6 3 1",
    "mumax": 2,
    "nmaxbase": 6,
    "type": "W Be",
    "lmin": "0 1 1",
    "bzeroflag": 0
    },
"CALCULATOR":
    {
    "calculator": "LAMMPSPACE",
    "energy": 1,
    "force": 1,
    "stress": 0
    },
"ESHIFT":
    {
    "W": 0.0,
    "B": 0.0
    },
"SOLVER":
    {
    "solver": "RIDGE",
    "compute_testerrs": 1,
    "detailed_errors": 1
    },
"RIDGE":
    {
    "local_solver": 1,
    "alpha": 1e-4
    },
"SCRAPER":
    {
    "scraper": "JSON" 
    },
"PATH":
    {
    "dataPath": "/Users/adrohsk/FitSNAP/examples/WBe_PRB2019/JSON"
    },
"OUTFILE":
    {
    "output_style": "PACE",
    "metrics": "WBe_metrics.md",
    "potential": "WBe_pot"
    },
"REFERENCE":
    {
    "units": "metal",
    "atom_style": "atomic",
    "pair_style": "zero 6.0",
    "pair_coeff": "* *"
    },
"EXTRAS":
    {
    "dump_descriptors": 0,
    "dump_truth": 0,
    "dump_weights": 0,
    "dump_dataframe": 0
    },
"GROUPS":
    {
    "group_sections": "name training_size testing_size eweight fweight",
    "group_types": "str float float float float",
    "smartweights": 0,
    "random_sampling": 0,
    "DFT_MD_1000K":     "0.03      0.0      1e-1      1.0",
    "DFT_MD_300K":      "0.03      0.0      1e-1      1.0",
    "EOS_BCC":          "0.01      0.0      1e-1      1.0",
    "Elast_BCC_Shear":  "0.01      0.0      1e-1      1.0",
    "Elast_BCC_Vol":    "0.01      0.0      1e-1      1.0"
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
print(f">>> {len(fs.data)} configs")
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
np.savetxt("amat_full.dat", amat)

"""
print(np.shape(amat))
#n = np.shape(amat)[0]
#reg = np.identity(n)*1e-12
#mat = amat + reg
cond = LA.cond(amat)/1e9

print(cond)
"""



