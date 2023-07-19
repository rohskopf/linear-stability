"""
Python script using library API to:
- Loop over all configurations in parallel and perform the transpose trick C = A^T * A and d = A^T * b.
- Sum these `C` and `d` arrays for all configurations.
- Perform least squares (or ridge regression) fit.

Usage:

    mpirun -np P python example.py

Afterwards, use the `in.run` LAMMPS script to run MD with:

    mpirun -np P lmp -in in.run

NOTE: This workflow is under development and therefore script requires changes.

- `settings` variable can be a dictionary like the example provided, or path to a fitsnap input script.
- `alval`: Ridge regression regularization parameter.
- Comment or uncomment `least_squares` or `ridge` at end of script to choose fitting method.

"""

from mpi4py import MPI
from fitsnap3lib.fitsnap import FitSnap
import numpy as np
from scipy.linalg import lstsq
from sys import float_info as fi
from sklearn.linear_model import Ridge

def least_squares(c, d):
    """
    Normal least squares fit.
    """
    coeffs, residues, rank, s = lstsq(c, d, 1.0e-13)
    return coeffs

def ridge(c, d):
    """
    Least squares fit with ridge regularization.
    """
    alval = 1.e-4
    reg = Ridge(alpha = alval, fit_intercept = False)
    reg.fit(c, d)
    return reg.coef_.T # return transpose if using sklearn ridge

def error_analysis(instance):
    """
    Calculate errors associated with a fitsnap instance that does not have shared arrays for the 
    entire A matrix or b vector, e.g. like we have when doing transpose trick. Here we loop over 
    all configurations and accumulate errors one at a time.

    Args:
        instance: fitsnap instance that contains a valid `fit`.

    Prints total MAE for all configurations in the data set.
    TODO: Organize this to calculate group errors or other kinds of errors.
    """

    # Get total number of atoms and configs across all procs for calculating average errors.
    nconfigs_all = len(instance.pt.shared_arrays["number_of_atoms"].array)
    natoms_all = instance.pt.shared_arrays["number_of_atoms"].array.sum()

    energy_mae = 0.0
    force_mae = 0.0
    stress_mae = 0.0
    for i, configuration in enumerate(instance.data):
        # TODO: Add option to print descriptor calculation progress on single proc.
        if (i % 10 == 0):
            fs.pt.single_print(i)
        a,b,w = instance.calculator.process_single(configuration, i)
        aw, bw = w[:, np.newaxis] * a, w * b
        
        pred = a @ coeffs

        # Energy error.
        energy = pred[0,0]
        ediff = np.abs(energy-b[0])
        energy_mae += ediff/nconfigs_all

        # Force error.
        ndim_force = 3
        nrows_force = ndim_force * configuration['NumAtoms']
        force = pred[1:nrows_force+1,0]
        force_mae += np.sum(abs(force-b[1:nrows_force+1]))/(3*natoms_all)

        # Stress error.
        stress = pred[-6:,0]
        sdiff_mae = np.mean(np.abs(stress-b[-6:]))
        stress_mae += sdiff_mae/nconfigs_all
    # Good practice after a large parallel operation is to impose a barrier to wait for all procs to complete.
    instance.pt.all_barrier()

    # Reduce errors across procs.
    energy_mae = np.array([energy_mae])
    force_mae = np.array([force_mae])
    stress_mae = np.array([stress_mae])
    energy_mae_all = np.array([0.0])
    force_mae_all = np.array([0.0])
    stress_mae_all = np.array([0.0])
    comm.Allreduce([energy_mae, MPI.DOUBLE], [energy_mae_all, MPI.DOUBLE])
    comm.Allreduce([force_mae, MPI.DOUBLE], [force_mae_all, MPI.DOUBLE])
    comm.Allreduce([stress_mae, MPI.DOUBLE], [stress_mae_all, MPI.DOUBLE])

    # Print errors.
    if (rank==0):
        print(energy_mae_all[0])
        print(force_mae_all[0])
        print(stress_mae_all[0])


# Declare a communicator (this can be a custom communicator as well).
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()

# Create an input dictionary containing settings.
settings = \
{
"ACE":
    {
    "numTypes": 2,
    "rcutfac": "5.790  5.007  5.007  4.224",
    "lambda": "1.737  1.502  1.502  1.267",
    "rcinner": "1.705  1.403  1.403  1.100",
    "drcinner": "0.01 0.01 0.01 0.01",
    "ranks": "1 2 3",
    "lmax":  "1 2 3",
    "nmax": "8 3 1",
    "mumax": 2,
    "nmaxbase": 8,
    "type": "In P",
    "lmin": "0 0 0",
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
    "In": -1.65967588701534,
    "P": 4.38159549501534
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
    "dataPath": "../../InP_JPCA2020/JSON"
    },
"OUTFILE":
    {
    "output_style": "PACE",
    "metrics": "InP_metrics.md",
    "potential": "InP_pot"
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
    "group_sections": "name training_size testing_size eweight fweight vweight",
    "group_types": "str float float float float float",
    "smartweights": 0,
    "random_sampling": 1,
    "aa":     "0.8      0.2      1.00000000      100.00000000      1.E-12",
    "aIn":     "0.8      0.2      1.00000000      100.00000000      1.E-12",
    "aP":     "0.8      0.2      1.00000000      100.00000000      1.E-12",
    "Bulk":     "0.8      0.2      1.00000000      0.00100000      1.E-12",
    "EOS":     "0.8      0.2      40.00000000      100.00000000      1.E-12",
    "iIn":     "0.8      0.2      2.00000000      100.00000000      1.E-12",
    "iP":     "0.8      0.2      2.00000000      100.00000000      1.E-12",
    "s_aa":    "0.8      0.2      10.00000000      100.00000000      1.E-12",
    "s_aIn":     "0.8      0.2      10.00000000      100.00000000      1.E-12",
    "s_aP":     "0.8      0.2      10.00000000      200.00000000      1.E-12",
    "Shear":     "0.8      0.2      22.00000000      100.00000000      1.E-12",
    "s_iIn":     "0.8      0.2      10.00000000      300.00000000      1.E-12",
    "s_iP":     "0.8      0.2      10.00000000      300.00000000      1.E-12",
    "Strain":     "0.8      0.2      13.00000000      0.00100000      1.E-12",
    "s_vIn":    "0.8      0.2      10.00000000      100.00000000      1.E-12",
    "s_vP":     "0.8      0.2      1.00000000      100.00000000      1.E-12",
    "s_vv":     "0.8      0.2      1.00000000      100.00000000      1.E-12",
    "vP":     "0.8      0.2      1.00000000      100.00000000      1.E-12"
    },
"MEMORY":
    {
    "override": 0
    }
}

# Alternatively, settings could be provided in a traditional input file:
#settings = "../../Ta_Linear_JCP2014/Ta-example.in"

# Create a FitSnap instance using the communicator and settings:
fs = FitSnap(settings, comm=comm, arglist=["--overwrite"])

# Scrape configurations to create and populate the `snap.data` list of dictionaries with structural info.
fs.scrape_configs()

# Allocate `C` and `d` fitting arrays.
a_width = fs.calculator.get_width()
c = np.zeros((a_width,a_width)) # This will also include weights.
d = np.zeros((a_width,1))

# Create fitsnap dictionaries (optional if you want access to distributed lists of groups, etc.)
fs.calculator.create_dicts(len(fs.data))
# Allocate per-config data like natoms per config, useful for error analysis.
fs.calculator.allocate_per_config(fs.data)
# Create `C` and `d` arrays for solving lstsq with transpose trick.
a_width = fs.calculator.get_width()
c = np.zeros((a_width,a_width))
d = np.zeros((a_width,1))
c_all = np.zeros((a_width,a_width))
d_all = np.zeros((a_width,1))
if rank==0:
    print(f">>> Looping over {len(fs.data)} structures.")
for i, configuration in enumerate(fs.data):
    # TODO: Add option to print descriptor calculation progress on single proc.
    if (i % 10 == 0):
        fs.pt.single_print(i)
    a,b,w = fs.calculator.process_single(configuration, i)
    aw, bw = w[:, np.newaxis] * a, w * b

    cm = np.matmul(np.transpose(aw), aw)
    dm = np.matmul(np.transpose(aw), bw[:,np.newaxis])
    c += cm
    d += dm
# Good practice after a large parallel operation is to impose a barrier to wait for all procs to complete.
fs.pt.all_barrier()

# Reduce C and D arrays across procs.
comm.Allreduce([c, MPI.DOUBLE], [c_all, MPI.DOUBLE])
comm.Allreduce([d, MPI.DOUBLE], [d_all, MPI.DOUBLE])

# Perform least squares fit.
#coeffs = least_squares(c_all,d_all)
coeffs = ridge(c_all, d_all)
# Now `coeffs` is owned by all procs, good for parallel error analysis.

# Calculate errors for this instance (not required).
error_analysis(fs)

# Write LAMMPS files.
# NOTE: Without error analysis, `fitsnap.solver.errors` is an empty list and will not be written to file.
fs.output.output(coeffs, fs.solver.errors)
