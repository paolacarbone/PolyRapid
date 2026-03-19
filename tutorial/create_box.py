#!/usr/bin/env python3
import subprocess
import glob
import os
import shutil

# set target box size and atom limit
box_size = 25.0
atom_limit = 50000


for subdir in sorted(glob.glob("W01*")):
    if not os.path.isdir(subdir):
        continue

    print(f"\nEntering directory: {subdir}")
    gro_files = glob.glob(os.path.join(subdir, "*.gro"))

    if not gro_files:
        print("  No .gro files found, skipping.")
        continue
    gro_file = os.path.basename(gro_files[0])
    
    # run a gromacs minimisation because we need to ensure that the chains are not too sterically hindered. 
    enmin_template = """integrator  = steep
    emtol          = 500
    emstep         = 0.01
    nsteps         = 500000
    nstenergy      = 500
    nstlog         = 500

    ; Change this to XTC
    xtc-grps                = System   ; Define the groups for XTC output

    ; Parameters describing how to find the neighbors of each atom and how to calculate the interactions
    nstlist         = 1
    cutoff-scheme   = Verlet
    ns_type         = grid
    coulombtype     = PME
    rcoulomb        = 1.0
    rvdw            = 1.0
    pbc             = xyz
    """

    # Write to file in the current working directory
    mdp_path = os.path.join(subdir, "enmin.mdp")
    with open(mdp_path, "w") as f:
        f.write(enmin_template)
    cmd = [
        "gmx", "grompp",
        "-f", "enmin.mdp",
        "-c", gro_file,
        "-p", "gromacs.top",
        "-o", "enmin.tpr"
    ]
    subprocess.run(cmd,cwd=subdir,check=True)
    cmd = [
        "gmx", "mdrun",
        "-deffnm", "enmin"
    ]
    subprocess.run(cmd,cwd=subdir,check=True)

    # rename the old file now we have enmin.gro
    shutil.move(
        os.path.join(subdir,gro_file),
        os.path.join(subdir,f"{gro_file}.oldgro")
    )

    # use the minimised chain for the rest of the insertion
    gro_files = [os.path.join(subdir, "enmin.gro")]

    for gro_file in gro_files:
        print(f"  Processing {gro_file}...")

        # Count number of atoms (2nd line in .gro file)
        with open(gro_file, "r") as f:
            lines = f.readlines()
        natoms = int(lines[1].strip())

        # Compute repeats
        nrepeat = atom_limit // natoms
        if nrepeat <= 0:
            print(f"    Skipping {gro_file}: too many atoms ({natoms}) for {atom_limit} limit.")
            continue

        print(f"    Number of atoms: {natoms}")
        print(f"    Repeats (atom_limit // {natoms}): {nrepeat}")

        # Define output file
        base = os.path.splitext(gro_file)[0]
        out_file = f"{base}_box.gro"

        # Build gmx insert-molecules command
        cmd = [
            "gmx_mpi", "insert-molecules",
            "-ci", gro_file,
            "-nmol", str(nrepeat),
            "-box", str(box_size), str(box_size), str(box_size),
            "-try", "100000",
            "-o", out_file
        ]

        print("    Running:", " ".join(cmd))
        subprocess.run(cmd,check=True)

        # Rename original file to .oldgro
        oldgro_file = f"{base}.oldgro"
        shutil.move(gro_file, oldgro_file)
        print(f"    Renamed {gro_file} -> {oldgro_file}")

        # Update topology file if it exists
        top_files = glob.glob(os.path.join(subdir, "*.top"))
        if top_files:
            top_file = top_files[0]  # assume only one .top per directory
            print(f"    Updating topology file: {top_file}")

            with open(top_file, "r") as f:
                lines = f.readlines()

            updated_lines = []
            for line in lines:
                parts = line.split()
                # Look for a line like: XYZ   1
                if len(parts) == 2 and parts[1].isdigit():
                    molname, count = parts
                    if int(count) == 1:  # only replace if it was "1"
                        updated_line = f"{molname}\t{nrepeat}\n"
                        updated_lines.append(updated_line)
                        print(f"      Updated: {line.strip()} -> {updated_line.strip()}")
                        continue
                updated_lines.append(line)

            with open(top_file, "w") as f:
                f.writelines(updated_lines)
        else:
            print("    No .top file found in this directory.")

print("\nAll W01* directories processed.")
