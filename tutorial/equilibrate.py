import os
import subprocess
from pathlib import Path
import numpy as np
from scipy.integrate import simpson
import argparse
import pandas as pd
import shutil


def Check_gmx_command():
    # Check if gmx_mpi or gmx are available, ideally both. If not fall back to the available one
    has_gmx = shutil.which("gmx") is not None
    has_gmx_mpi = shutil.which("gmx_mpi") is not None
    if has_gmx and has_gmx_mpi:
        grompp_cmd = "gmx"
        mdrun_cmd = "gmx_mpi"
    elif has_gmx:
        grompp_cmd = "gmx"
        mdrun_cmd = "gmx"
        print("[WARNING] 'gmx_mpi' not found, falling back to 'gmx' for both grompp and mdrun.")
    elif has_gmx_mpi:
        grompp_cmd = "gmx_mpi"
        mdrun_cmd = "gmx_mpi"
        print("[WARNING] 'gmx' not found, falling back to 'gmx_mpi' for both grompp and mdrun.")
    else:
        raise EnvironmentError("Neither 'gmx' nor 'gmx_mpi' command found in PATH. Please install or load GROMACS.")
    return grompp_cmd, mdrun_cmd
    
def run(cmd, **kwargs):
    """
    Executes a command as a subprocess, prints the command being run, and checks for errors.
    Takes the errors_only keyword argument to suppress all outputs bar errors to the slurm file/console.
    Args:
        cmd (list): The command and its arguments to execute.
        **kwargs: Additional keyword arguments to pass to subprocess.run().

    Returns:
        subprocess.CompletedProcess: The result of the executed command.

    Raises:
        RuntimeError: If the command returns a non-zero exit code.
    """
    global ERRORS_ONLY
    print(f"Running: {' '.join(str(x) for x in cmd)}", flush=True)
    if ERRORS_ONLY:
        result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, **kwargs)
        if result.returncode != 0:
            print(result.stdout.decode(), flush=True)
            print(result.stderr.decode(), flush=True)
            raise RuntimeError(f"Command failed: {' '.join(str(x) for x in cmd)}")
    else: 
        # print everything as normal
        result = subprocess.run(cmd, **kwargs)
        if result.returncode != 0:
            raise RuntimeError(f"Command failed: {' '.join(str(x) for x in cmd)}")

def compute_delta_rdf(data_file1, data_file2, interval_length=2.0):
    """
    Compute the (normalized) integrated absolute difference between two radial distribution functions (RDFs).
    Loads both the rdf data files in format (r, g(r)). Interpolates the second dataset onto the r-grid of the first dataset
    if the r-values differ. Computes the pointwise absolute difference of the RDFs, integrates that absolute difference from r=0 up to interval_length (r = 2)
    using Simpson's rule, and returns the result divided by 2.0 (normalization).

    Parameters
    ----------
    data_file1 : str or path-like
        Path to the first data file. Read with np.loadtxt ignoring gromacs comments
    data_file2 : str or path-like
        Path to the second data file. Read with np.loadtxt ignoring gromacs comments
    interval_length : float, optional
        Upper integration limit for r (default is 2.0). Only r values <= interval_length are included in
        the Simpson integration.

    Returns
    -------
    float
        The Simpson-integrated absolute difference between the two RDFs over r <= interval_length,
        divided by 2.0 (normalization).
    """
    data1 = np.loadtxt(data_file1, comments=["@", "#"], usecols=(0, 1))
    data2 = np.loadtxt(data_file2, comments=["@", "#"], usecols=(0, 1))
    r_values = data1[:, 0]
    rdf1_values = data1[:, 1]
    rdf2_values = np.interp(r_values, data2[:, 0], data2[:, 1]) # we interpolate to the same r values but there should never be a difference in r values, so the data will be the same. This is a failsafe.
    absolute_delta_rdf = np.abs(rdf1_values - rdf2_values)
    mask = r_values <= interval_length
    integrated_absolute_delta_rdf = simpson(absolute_delta_rdf[mask], r_values[mask])
    return integrated_absolute_delta_rdf/2.0 # Normalized by cut off

def compute_delta_polystat(polydata1, polydata2):
    """
    Compute absolute mean differences between paired data files for polydata.

    Parameters
    ----------
    polydata1 : str
        Path to the first polydata file.
    polydata2 : str
        Path to the second polydata file.
    Returns
    -------
    tuple of float
        A 4-tuple containing the absolute mean differences in the following order:
        (absolute_mean_E2E, absolute_mean_rg)
        - absolute_mean_E2E: absolute value of the mean of (polydata2[:,0] - polydata1[:,0])
        - absolute_mean_rg:  absolute value of the mean of (polydata2[:,1] - polydata1[:,1])
    """
    # Polydata: columns 1 and 2 (index 1 and 2)
    data1 = np.loadtxt(polydata1, comments=["@", "#"], usecols=(1, 2))
    data2 = np.loadtxt(polydata2, comments=["@", "#"], usecols=(1, 2))
    min_len = min(len(data1), len(data2))
    data1 = data1[:min_len]
    data2 = data2[:min_len]
    diff_col1 = data2[:, 0] - data1[:, 0]
    diff_col2 = data2[:, 1] - data1[:, 1]
    absolute_mean_E2E = np.abs(np.mean(diff_col1))
    absolute_mean_rg = np.abs(np.mean(diff_col2))

    return absolute_mean_E2E, absolute_mean_rg

def create_random_index(gro_file, tpr_file, index_file, grompp_kw, fraction=0.1):
    """
    Create a GROMACS index file by randomly selecting a fraction of residues from a .gro file.

    Randomly pick a fraction of the residues in the GRO file and create and index that has two groups
    one with the carbon atoms of selected residues and one with the carbon atoms of all other residues in the box
    Uses gmx select: https://manual.gromacs.org/2024.3/onlinehelp/gmx-select.html
    Produces an index file that is used for the RDF calculations.

    Parameters
    ----------
    gro_file : str or pathlib.Path
        Path to the input GROMACS .gro file. The function reads the atom lines and expects
        residue numbers to occupy the beginning of each atom line (parsed as the first 5
        characters, converted to int).
    tpr_file : str or pathlib.Path
        Path to the corresponding GROMACS .tpr file used by `gmx select` (supplies topology
        and coordinate metadata required by the selection tool).
    index_file : str or pathlib.Path
        Path where the output GROMACS index (.ndx) file will be written.
    fraction : float, optional
        Fraction of unique residues to select (range 0.0–1.0). The number of selected residues
        is computed as max(1, int(n_residues * fraction)), so at least one residue will be
        selected even if fraction is zero or very small. Default is 0.1.
    """
    with open(gro_file) as f:
        lines = f.readlines()[2:-1]
    resids = [int(line[:5]) for line in lines]
    unique_resids = np.unique(resids)
    n_mol = len(unique_resids)
    n_sel = max(1, int(n_mol * fraction))
    selected = set(np.random.choice(unique_resids, n_sel, replace=False))

    sel1 = '(' + ' or '.join(f"resnr {i}" for i in sorted(selected)) + ') and mass == 12.0115'
    sel2 = f"(not ({' or '.join(f'resnr {i}' for i in sorted(selected))})) and mass == 12.0115"
    sel3 = 'mass > 2.0'
    selection = f"{sel1}; {sel2} ; {sel3}"

    run([
        grompp_kw, "select",
        "-f", str(gro_file),
        "-s", str(tpr_file),
        "-select", selection,
        "-on", str(index_file)
    ])

def main():
    """
    Automated GROMACS equilibration workflow.
    Utilising some helper functions and GROMACS commands the multi-step equilibration procedure runs here
    Parses two arguments from the command line, checks for required files in the current working directory,
    runs energy minimisation followed by repeated NPT + annealing cycles, computes summary statistics and
    convergence metrics writing these to a csv file per step and a final csv file containing information 
    from all steps.
    - Parse CLI arguments:
        --deltardf (float, default=0.01): convergence threshold for normalized integrated
            absolute delta RDF (0..2 nm).
        --maxcycles (int, default=6): maximum number of cycles to attempt.
    - Ensures the required files are present in the current working directory:
        npt.mdp, anneal.mdp, enmin.mdp, one *.gro, one *.top
        If any are missing, the function prints an error and exits.
    - Parse annealing_time from anneal.mdp this is used so that we get statistics only over the last 5 ns of annealing.    
        If not present, raises ValueError.
    - Perform energy minimization:
        - Create "Enmin" directory, grompp and run energy minimisation.
        - Uses outputs: 'enmin.gro' and 'enmin.tpr' as starting point for index creation.
    - Create an index file for RDF calculations using:
        - create_random_index(enmin.gro, enmin.tpr, rdf_groups.ndx).
    - For each step in 1..maxcycles:
        - Create directories Step{N}/npt and Step{N}/anneal.
        - Choose input .gro: enmin.gro for step 1, otherwise the previous step's
          anneal gro file (Step{N-1}/anneal/step{N-1}_anneal.gro).
        - Run high pressure NPT to push the system out of a steady state
        - Run simulated annealing to relax the system at constant pressure. (1atm)
        - Compute average temperature and density over the final annealing period
            - GMX energy over the final 5 ns of annealing
            - Ensures the density and temperature were sensible and within a good std dev.
            - sets the values to 0 if not possible to extract (this is a placeholder)
        - Compute RDF (gmx rdf) using rdf_groups.ndx and rmax=2.002 nm on the annealing simulation.
        - Run gmx polystat to extract polymer statistics (E2E, Rg) over the last 5 ns.
        - If step >= 2:
            - Compute delta RDF between Step{N-1}/anneal/rdf_step{N-1}_anneal.xvg
              and Step{N}/anneal/rdf_step{N}_anneal.xvg using compute_delta_rdf().
        - Check the equilibration status by comparing delta RDF to the threshold:
            - Print "Box is equilibrated", write All_properties.csv and All_convergence.csv,
              and break early (successful convergence).
    - If the loop completes without meeting the convergence criterion, print a warning
      that the maximum number of cycles was reached and the box may not be equilibrated.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--deltardf", type=float, default=0.01)
    parser.add_argument("--maxcycles", type=int, default=6)
    parser.add_argument("--errors_only", action="store_true", help="If set, only error messages will be printed to the console/slurm output.")
    args = parser.parse_args()
    global ERRORS_ONLY
    ERRORS_ONLY = args.errors_only

    BASE_DIR = Path.cwd()
    npt_mdpfile = next(BASE_DIR.glob("npt.mdp"), None)
    anneal_mdpfile = next(BASE_DIR.glob("anneal.mdp"), None)
    enmin_mdpfile = next(BASE_DIR.glob("enmin.mdp"), None)
    gro_file = next(BASE_DIR.glob("*.gro"), None)
    top_file = next(BASE_DIR.glob("*.top"), None)
    if not all([npt_mdpfile, anneal_mdpfile, enmin_mdpfile, gro_file, top_file]):
        print("Missing files. Check that you have npt.mdp, anneal.mdp, *.gro and *.top in the base directory.", flush=True)
        exit(1)
    # Added logic to check the gmx/gmx_mpi commands
    grompp_kw, mdrun_kw = Check_gmx_command()

    #  Find annealing_time from anneal.mdp 
    with open(anneal_mdpfile, "r") as f:
        annealing_time = None
        for line in f:
            if "annealing_time" in line:
                # Extract the last number from the line
                tokens = line.split()
                numbers = [float(tok) for tok in tokens if tok.replace('.', '', 1).isdigit()]
                if numbers:
                    annealing_time = numbers[-1]
    if annealing_time is None:
        raise ValueError("Could not find 'annealing_time' in anneal.mdp")
   
    #  Energy Minimisation Step 
    enmin_dir = BASE_DIR / "Enmin"
    enmin_dir.mkdir(parents=True, exist_ok=True)
    run([
        grompp_kw, "grompp",
        "-f", str(enmin_mdpfile),
        "-c", str(gro_file),
        "-p", str(top_file),
        "-o", str(enmin_dir / "enmin.tpr"),
    ])
    if mdrun_kw == "gmx_mpi":
        run([
            "mpirun", mdrun_kw, "mdrun",
            "-deffnm", str(enmin_dir / "enmin")
        ])
    else:
        run([
            mdrun_kw, "mdrun", "-deffnm", str(enmin_dir / "enmin")
        ])

    # Create index file for RDF calculations 
    index_file = BASE_DIR / "rdf_groups.ndx"
    gro_file = enmin_dir / "enmin.gro"
    tpr_file = enmin_dir / "enmin.tpr"
    create_random_index(gro_file, tpr_file, index_file, grompp_kw)

    # Start from step 1, up to maxcycles
    converged = False
    all_df = pd.DataFrame()
    all_convergence = pd.DataFrame()
    for step in range(1, args.maxcycles + 1):
        step_npt = BASE_DIR / f"Step{step}/npt"
        step_anneal = BASE_DIR / f"Step{step}/anneal"
        step_npt.mkdir(parents=True, exist_ok=True)
        step_anneal.mkdir(parents=True, exist_ok=True)

        # Input gro for NPT
        if step == 1:
            input_gro = gro_file
        else:
            input_gro = BASE_DIR / f"Step{step-1}/anneal/step{step-1}_anneal.gro"

        # NPT
        run([grompp_kw ,"grompp", "-f", str(npt_mdpfile), "-c", str(input_gro), "-p", str(top_file), "-o", str(step_npt / f"step{step}_npt.tpr")])
        if mdrun_kw == "gmx_mpi":
            run([
                "mpirun", mdrun_kw, "mdrun",
                "-deffnm", str(step_npt / f"step{step}_npt")
            ])
        else:
            run([
                mdrun_kw, "mdrun", "-deffnm", str(step_npt / f"step{step}_npt")
            ])

        # Anneal
        run([grompp_kw, "grompp", "-f", str(anneal_mdpfile), "-c", str(step_npt / f"step{step}_npt.gro"), "-p", str(top_file), "-o", str(step_anneal / f"step{step}_anneal.tpr")])
        if mdrun_kw == "gmx_mpi":
            run([
                "mpirun", mdrun_kw, "mdrun",
                "-deffnm", str(step_anneal / f"step{step}_anneal")
            ])
        else:
            run([
                mdrun_kw, "mdrun", "-deffnm", str(step_anneal / f"step{step}_anneal")
            ])

        # Density + Temperature Average over last 5 ns
        run([grompp_kw , "energy", "-f", str(step_anneal / f"step{step}_anneal.edr"), "-o", str(step_anneal / f"temperature_density_step{step}_anneal.xvg"), "-b", str(annealing_time)], input=b"Temperature\nDensity\n")
        # Extract the gromacs data into an array
        data = np.loadtxt(step_anneal / f"temperature_density_step{step}_anneal.xvg", comments=["@", "#"], usecols=(1, 2))
        if data.size > 0:
            avg_temperature, avg_density = np.mean(data[:,0]) , np.mean(data[:,1])
        else: 
            avg_temperature, avg_density = 0, 0

        # RDF Anneal
        run([grompp_kw, "rdf", "-f", str(step_anneal / f"step{step}_anneal.xtc"), "-s", 
             str(step_anneal / f"step{step}_anneal.tpr"),"-n", str(index_file), "-b", str(annealing_time), "-rmax", "2.002",  "-o", 
             str(step_anneal / f"rdf_step{step}_anneal.xvg")], input=b"0\n1\n") #has to be 2.002 or we actually get 1.998 nm - the difference is trivial but exists.

        # GMX polystat
        run([grompp_kw, "polystat", "-f", str(step_anneal / f"step{step}_anneal.xtc"),
             "-s", str(step_anneal / f"step{step}_anneal.tpr"),
             "-n", str(index_file),
             "-o", str(step_anneal / f"polystat_step{step}_anneal.xvg")], 
            input=b"2\n")

        # Step 2 check
        if step >= 2:
            prev = step - 1
            rdf1 = BASE_DIR / f"Step{prev}/anneal/rdf_step{prev}_anneal.xvg"
            rdf2 = BASE_DIR / f"Step{step}/anneal/rdf_step{step}_anneal.xvg"
            deltardf = compute_delta_rdf(str(rdf1), str(rdf2))
            print(f"Normalized integrated absolute delta RDF over 0 to 2 nm: {deltardf:.4f}",flush=True)

            #Data collection for Sam
            polydata1 = BASE_DIR / f"Step{prev}/anneal/polystat_step{prev}_anneal.xvg"
            polydata2 = BASE_DIR / f"Step{step}/anneal/polystat_step{step}_anneal.xvg"
            absE2E, absRg = compute_delta_polystat(str(polydata1), str(polydata2))
            df_convergence = pd.DataFrame([{
                "Step": f"{prev}->{step}",
                "Abs Mean E2E": absE2E,
                "Abs Mean Rg": absRg,
                "Delta RDF": deltardf
                }])
            convergence_csv = BASE_DIR / f"Step{step}/step{step}_convergence.csv"
            df_convergence.to_csv(convergence_csv, index=False, float_format="%.4f")
            print(df_convergence.to_string(index=False), flush=True)
            all_convergence = pd.concat([all_convergence, df_convergence], ignore_index=True)
        else:
            deltardf = np.nan
        
        # Print properties to .csv files per step
        poly_data = np.loadtxt(step_anneal / f"polystat_step{step}_anneal.xvg", comments=["@", "#"])

        row = {
            "Step": step,
            "Temperature": avg_temperature,
            "Density": avg_density,
            "Density Std": np.std(data[:,1]),
            "E2E Mean": np.mean(poly_data[:, 1]),
            "E2E Std": np.std(poly_data[:, 1]),
            "Rg Mean": np.mean(poly_data[:, 2]),
            "Rg Std": np.std(poly_data[:, 2]),
            "Delta RDF": deltardf,
        }
        df = pd.DataFrame([row])
        print(df.to_string(index=False), flush=True)
        step_csv = BASE_DIR / f"Step{step}/step{step}_properties.csv"
        df.to_csv(step_csv, index=False, float_format="%.4f")

        # Create total dataframes
        all_df = pd.concat([all_df, df], ignore_index=True)          
            
        if step>=2 and deltardf < args.deltardf:
            print("Box is equilibrated", flush=True)
            converged = True
            all_df.to_csv(BASE_DIR / "All_properties.csv", index=False, float_format="%.4f")
            all_convergence.to_csv(BASE_DIR / "All_convergence.csv", index=False, float_format="%.4f")
            break

    if not converged:
        print(f"Maximum number of cycles ({args.maxcycles}) reached. Box may not be equilibrated.", flush=True)
        all_df.to_csv(BASE_DIR / "All_properties.csv", index=False, float_format="%.4f")
        all_convergence.to_csv(BASE_DIR / "All_convergence.csv", index=False, float_format="%.4f")

if __name__ == "__main__":
    main()
