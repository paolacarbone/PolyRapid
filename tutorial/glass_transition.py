import subprocess
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import argparse
import os
from shutil import copyfile
import pandas as pd

def run(cmd, **kwargs):
    print(f"Running: {' '.join(str(x) for x in cmd)}")
    result = subprocess.run(cmd, **kwargs)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(str(x) for x in cmd)}")
    return result

def read_mdp(mdp_path):
    """Read an MDP file into a dict of key: value (as strings)."""
    options = {}
    with open(mdp_path, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith(";"):
                if "=" in line:
                    key, val = line.split("=", 1)
                    options[key.strip()] = val.strip()
    return options

def get_total_time_from_mdp(mdp_options):
    """
    Calculate total simulation time in ps from mdp options.
    Assumes nsteps and (dt or time unit) are present.
    """
    nsteps = int(mdp_options.get("nsteps", 0))
    dt = float(mdp_options.get("dt", 0.001))  # default GROMACS dt is 0.001 ps
    total_time = nsteps * dt
    return int(total_time)

def write_mdp(options, out_path):
    """Write an MDP options dict to a file."""
    with open(out_path, "w") as f:
        for key, val in options.items():
            f.write(f"{key} = {val}\n")


# Analysis functions 
def intersecting_lines(x, x0, y0, m1, m2):
    # Piecewise linear: before x0 use m1, after x0 use m2
    return np.where(x < x0, m1 * (x - x0) + y0, m2 * (x - x0) + y0)

def linear_fit(x, y, GuessTg):
    # Initial guesses: x0=GuessTg, y0=mean(y), m1 and m2 from np.polyfit
    mask_below = x < GuessTg
    mask_above = x >= GuessTg
    if np.sum(mask_below) < 2 or np.sum(mask_above) < 2:
        raise ValueError("Not enough data points below or above GuessTg for linear fit.")
    m1_init, b1_init = np.polyfit(x[mask_below], y[mask_below], 1)
    m2_init, b2_init = np.polyfit(x[mask_above], y[mask_above], 1)
    y0_init = (b1_init + b2_init) / 2
    p0 = [GuessTg, y0_init, m1_init, m2_init]

    popt, _ = curve_fit(intersecting_lines, x, y, p0=p0, maxfev=10000)
    x0, y0, m1, m2 = popt
    return x0, y0, m1, m2

def glass_transition_analysis(denstemp_path, GuessTg=200, RangeLim=400):
    Temperature, Density = np.loadtxt(denstemp_path, skiprows=0, comments=['#','@'], unpack=True, usecols=(1,2))

    # Apply temperature domain limit
    mask = Temperature <= RangeLim
    Temperature_fit = Temperature[mask]
    Density_fit = Density[mask]

    # Plot linear fit (piecewise) on another plot
    x0, y0, m1, m2 = linear_fit(Temperature_fit, Density_fit, GuessTg)
    plt.figure(figsize=(5,5))
    plt.scatter(Temperature_fit, Density_fit, label="Data")
    T1 = np.linspace(min(Temperature_fit), x0, 100)
    T2 = np.linspace(x0, max(Temperature_fit), 100)
    plt.plot(T1, m1*(T1-x0)+y0, 'b--', label="Linear fit (below Tg)")
    plt.plot(T2, m2*(T2-x0)+y0, 'g--', label="Linear fit (above Tg)")
    plt.plot([x0, x0], [min(Density_fit), y0], 'k:', label=f"Tg (linear) = {x0:.1f} K")
    plt.scatter([x0], [y0], color='k')
    plt.xlabel("Temperature / K")
    plt.ylabel("Density / kg m$^{-3}$")
    plt.legend()
    plt.savefig("DensTemp.png", format="png")
    plt.close()

    return {
        "x0": x0,
    }

def main():
    parser = argparse.ArgumentParser(description="Run glass transition simulation and analysis.")
    parser.add_argument("--GuessTg", type=float, default=200, help="Initial guess for glass transition temperature (default: 200)")
    parser.add_argument("--RangeLim", type=float, default=400, help="Temperature range limit for fitting (default: 400)")
    args = parser.parse_args()

    BASE_DIR = Path.cwd()
    glass_dir = BASE_DIR / "GlassTransition"
    cooling_dir = glass_dir / "Cooling"
    heating_dir = glass_dir / "Heating"
    analysis_dir = glass_dir / "Analysis"
    cooling_dir.mkdir(parents=True, exist_ok=True)
    heating_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # Find the last equilibrated .gro file (from last anneal step)
    anneal_dirs = sorted(BASE_DIR.glob("Step*/anneal"), key=lambda p: int(p.parts[-2][4:]))
    if not anneal_dirs:
        raise RuntimeError("No anneal directories found. Run equilibration first.")
    last_anneal = anneal_dirs[-1]
    gro_file = next(last_anneal.glob("*.gro"), None)
    top_file = next(BASE_DIR.glob("*.top"), None)
    mdp_file = BASE_DIR / "glasstransition.mdp"

    if not gro_file or not top_file or not mdp_file.exists():
        raise RuntimeError("Missing glasstransition.mdp, .gro, or .top file.")

    abs_gro = gro_file.resolve()
    abs_top = top_file.resolve()
    abs_mdp = mdp_file.resolve()

    os.chdir(cooling_dir)
    run([
        "gmx", "grompp",
        "-f", str(abs_mdp),
        "-c", str(abs_gro),
        "-p", str(abs_top),
        "-o", "glass.tpr",
    ])
    run(["mpirun", "gmx_mpi", "mdrun", "-deffnm", "glass"])
    with open("denstemp.xvg", "w") as out:
        run(
            ["gmx", "energy", "-f", "glass.edr", "-o", "denstemp.xvg"],
            input=b"Temperature\nDensity\n",
            stdout=out
        )
    # perform some rough guess with the density-temperature data with the limits and guess imposed. It will be re-ran again after the initial guess with the whole range included.
    cooling_results = glass_transition_analysis("denstemp.xvg", GuessTg=args.GuessTg, RangeLim=800)

    # Move back to the base directory
    os.chdir(BASE_DIR)


    heating_mdp = heating_dir / "heating.mdp"
    copyfile(abs_mdp, heating_mdp)

    # Change the max temperature in the second run to be the Tg result + 150 K rounding to the nearest multiple of 50 K
    # Cooling rate is 20 K ns^-1 
    # Read the options from the existing mdp file
    mdp_options = read_mdp(heating_mdp)
    t_start = 0 

    # modify t_end, work out npoints and the total time
    print(f"Using linear Tg of {cooling_results['x0']:.2f} K for heating ramp.")
    t_end = 50 * round(((cooling_results['x0'] + 100)/50)) # This can be modified based on cooling rate in the future and the rounding window can be changed
    RangeLim, GuessTg = t_end,cooling_results['x0'] # we update the RangeLim  and GuessTg variables to use in analysis
    npoints = int(t_end / 50 ) + 1 # again can change this depending on the given range we want  --- we add 1 here so we have the correct number of points when 0 is included. 
    total_time = (npoints -1) * 2500 # cooling rate of 20 K ns^-1 in K ps^-1 we don't need the simulation to run long 

    # Use the total time to get the annealing temperature and time points 
    annealing_time_values = np.linspace(0,total_time, npoints,dtype=int) # we use npoints +1 to ensure we get some time at 0 K to ramp up from rather than just going nuts.
    annealing_temp_values = np.linspace(t_start,t_end,npoints,dtype=int) # as above
    annealing_time = " ".join(str(int(x)) for x in annealing_time_values)
    annealing_temp = " ".join(str(int(x)) for x in annealing_temp_values)

    # update the mdp file and run the heating loop
    mdp_options["annealing_npoints"] = npoints
    mdp_options["annealing_time"] = annealing_time
    mdp_options["annealing_temp"] = annealing_temp

    # Update nsteps to match the last annealing_time divided by dt
    dt = float(mdp_options.get("dt", 0.001))
    mdp_options["nsteps"] = str(int(total_time / dt))
    write_mdp(mdp_options, heating_mdp)
        
    os.chdir(heating_dir)
    run([
        "gmx", "grompp",
        "-f", str(heating_mdp),
        "-c", str((cooling_dir / "glass.gro").resolve()),
        "-p", str(abs_top),
        "-o", "glass.tpr",
    ])
    run(["mpirun", "gmx_mpi", "mdrun", "-deffnm", "glass"])
    with open("denstemp.xvg", "w") as out:
        run(
            ["gmx", "energy", "-f", "glass.edr", "-o", "denstemp.xvg"],
            input=b"Temperature\nDensity\n",
            stdout=out
        )
    heating_results = glass_transition_analysis("denstemp.xvg", GuessTg=GuessTg, RangeLim=RangeLim)
    # change back to the cooling directory to get the re-fit of the data
    os.chdir(cooling_dir)
    cooling_results = glass_transition_analysis("denstemp.xvg", GuessTg=GuessTg, RangeLim=RangeLim)
    # back to base directory
    os.chdir(BASE_DIR)

    # Copy denstemp.xvg files to analysis directory
    cooling_denstemp = cooling_dir / "denstemp.xvg"
    heating_denstemp = heating_dir / "denstemp.xvg"
    analysis_cooling = analysis_dir / "cooling_denstemp.xvg"
    analysis_heating = analysis_dir / "heating_denstemp.xvg"
    copyfile(cooling_denstemp, analysis_cooling)
    copyfile(heating_denstemp, analysis_heating)

    # Load data
    T_c, D_c = np.loadtxt(analysis_cooling, skiprows=0, comments=['#','@'], unpack=True, usecols=(1,2))
    T_h, D_h = np.loadtxt(analysis_heating, skiprows=0, comments=['#','@'], unpack=True, usecols=(1,2))

    # Apply temperature domain limit
    mask_c = T_c <= RangeLim
    mask_h = T_h <= RangeLim
    T_c_fit = T_c[mask_c]
    T_h_fit = T_h[mask_h]
    D_c_fit = D_c[mask_c]
    D_h_fit = D_h[mask_h]

    # Plot both cooling and heating stages
    plt.figure(figsize=(5,5))
    plt.scatter(T_c_fit, D_c_fit, label="Cooling Data")
    plt.scatter(T_h_fit, D_h_fit, label="Heating Data")

    # Linear fitting 
    x0_c, y0_c, m1_c, m2_c = linear_fit(T_c_fit, D_c_fit, GuessTg)
    x0_h, y0_h, m1_h, m2_h = linear_fit(T_h_fit, D_h_fit, GuessTg)
    T1c = np.linspace(min(T_c_fit), x0_c, 100)
    T2c = np.linspace(x0_c, max(T_c_fit), 100)
    T1h = np.linspace(min(T_h_fit), x0_h, 100)
    T2h = np.linspace(x0_h, max(T_h_fit), 100)
    plt.plot(T1c, m1_c*(T1c-x0_c)+y0_c, 'b--', label="Cooling fit (below Tg)")
    plt.plot(T2c, m2_c*(T2c-x0_c)+y0_c, 'g--', label="Cooling fit (above Tg)")
    plt.plot(T1h, m1_h*(T1h-x0_h)+y0_h, 'r--', label="Heating fit (below Tg)")
    plt.plot(T2h, m2_h*(T2h-x0_h)+y0_h, 'm--', label="Heating fit (above Tg)")
    plt.plot([x0_c, x0_c], [min(D_c_fit), y0_c], 'b:', label=f"Cooling Tg (linear) = {x0_c:.1f} K")
    plt.plot([x0_h, x0_h], [min(D_h_fit), y0_h], 'r:', label=f"Heating Tg (linear) = {x0_h:.1f} K")
    plt.scatter([x0_c], [y0_c], color='k')
    plt.scatter([x0_h], [y0_h], color='k')
    plt.xlabel("Temperature / K")
    plt.ylabel("Density / kg m$^{-3}$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(analysis_dir / "TgErrorPredict.png")
    plt.close()

    row = { 
        "Cooling_Tg_linear": x0_c,
        "Heating_Tg_linear": x0_h,
        "Mean Tg_linear": (x0_c + x0_h)/2,
    }
    results_df = pd.DataFrame([row])
    results_df.to_csv(analysis_dir / "Tg_results.csv", index=False, float_format="%.4f")
    with open(analysis_dir / "Tg_results.txt", "w") as f:
        # Linear
        f.write(f"Tg from cooling (linear): {x0_c:.2f}\n")
        f.write(f"Tg from heating (linear): {x0_h:.2f}\n")
        f.write(f"Tg mean (linear): {(x0_c + x0_h)/2:.2f}\n")

print("Glass transition analysis complete. Results and plot are in the Analysis directory.")
if __name__ == "__main__":
    main()
