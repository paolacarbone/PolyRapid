# PolyRapid
PolyRapid is an automated procedure for MD homopolymer equilibration from monomer SMILES. It proceeds in the following steps:

1. Create a single polymer chain XYZ from monomer SMILES using `create_chain.py` (default is 20mer). This includes a short geometry optimisation using Open Babel. 
2. Assign OPLS/AA forcefield parameters using DL_FIELD (`assign_forcefield.py`).
3. Create a ramdomly packed box (default 50,000 atoms) using `create_box.py`.
4. Equilibrate the box (`equilibrate.py`).
5. Optionally calculate the nematic order parameter (`order.py`) and glass transition temperature (`glass_transition.py`).

# Prerequisite and Dependencies

## Python

PolyRapid requires Python version 3.12 or higher and depends on the following libraries 

| Library | Version |
|----------|----------|
| mbuild   | 1.2.1   |
| openbabel   | 3.1.1   |
| rdkit   | 2025.09.1   |
| scipy   |  1.16.3  |
| numpy     |  2.2.6   |
| networkx   |  3.5   |
| matplotlib  | 3.10.6 |
| mdanalysis  | 2.9.0  |
| pandas   | 2.3.3  |

## Extra Software

To assign the forcefield, we require the `DL_FIELD` software (version 4.12), which can be downloaded from the following link: https://www.ccp5.ac.uk/dl_field-registration/. Follow the installation instructions 

To perform the MD equilibration, we require the MD package GROMACS (MPI and non-MPI, version 2023 or higher). To download from source, visit: https://manual.gromacs.org/documentation/2025.1/download.html

# Tutorial

In the following sections we provide a tutorial to equilibrate a box of polyethylene (20mer chains, 50,000 atoms).

## Setup

Install the required prerequisites and clone this repository via 

```
git clone https://github.com/paolacarbone/PolyRapid.git
```

move into the `tutorial` directory

```
cd tutorial/
```

## Creating a single chain 

The `library.csv` file provided by the tutorial contains a user-defined polymer ID (`PID`) and monomer SMILES string (`smiles_polymer`) for PE. The `*` symbol represents the connection points between monomers, and must be included in the monomer definition. To create the chain, ensure `library.csv` is in the current working directory and run:

```
python create_chain.py
```
This produces a new directory called `Output/` which contains folders named with the `PID` of each polymer in `library.csv`. Inside each folder is an XYZ coordinate file for a 20mer molecule. Note that the number of monomers per chain, input library file name and output directory name can be changed by modifying the following line in `create_chain.py`.

```
make_polymers("library.csv","Output/",20)
```

## Assigning the forcefield 

The file `assign_forcefield.py` provides a wrapper around the `DL_FIELD` software which assigns the OPLS/AA forcefield. Forcefield assignment details are provided using a `CONTROL` file, which we have provided an example of for convenience (`polymer.control`). For interested users, refer to the following `DL_FIELD` tutorials for beginners: https://lois181.github.io/dl_field_tutorials/. 

After installing and unpacking `DL_FIELD`, open `assign_forcefield.py` and provide the following file paths:

1. The path to the `DL_FIELD` executable (`dl_field`).
2. The path to the `CONTROL file.
3. The `Output/` directory where the PE XYZ is stored.

see below:

```
# Provide path to DL_FIELD root directory to initialise 
builder = TopologyBuilder(dl_field_path="/home/lois181/dl_f_4.12/") # path to dl_field exe

custom_control = "/home/lois181/dl_f_4.11/polymer.control" # control file path  
root_dir = "/home/lois181/code/try-chains-make/identifying-errors/Output" # directory where the xyz files are 
```
Ensure the `dl_field_wrapper` directory exists in the current working directory and run with:

```
python assign_forcefield.py
```
If successful, each folder in the `Output/` directory should now contain the GROMACS *.itp, *.gro and *.top files (there will also be an *mdp file which we will replace to perform the equilibration). 

## Create box 

We now use the PE *.gro file to create a PE box with ~50,000 atoms. Ensure your version of GROMACS has MPI support and run the following from the current working directory

```
python create_box.py
```
This performs a short minimisation of a single chain, before randomly packing the PE in a 25 x 25 x 25 nm^3 box, which is sufficient size for a 50,000 atom system for the majority of polymers. The output box is called `enmin_box.gro`. It also updates the topology file to reflect the correct number of residues. 

## Perform equilibration

The details of the equilibration procedure are outlined in https://arxiv.org/abs/2603.05362, for which we provide the GROMACS *.mdp files. Begin my moving into the `W01_P001/` directory which contains the PE files:

```
cd Output/W01_P001/
```

and copy `equilibrate.py` and *.mdp files:

```
cp ../../*mdp ./
cp ../../equilibrate.py ./
```

An example SLURM submission script is provided with `submit.slm`, which can be used from the `W01_P001/` directory. The threshold for equilibration, along with the maximum number of cycles can be controlled via

```
python equilibrate.py --deltardf 0.02 --maxcycles 6
```
Module loads and header lines should be changed to reflect your own HPC architecture. 

## Results

`equilibrate.py` automatically collects the polymer radius of gyration and end-to-end distance (via gmx polystat), and density (gmx energy) in the final 5ns of the last annealing cycle. The data can be found in the file `All_properties.csv` in the `W01_P001/` directory.

## Analyses 

We offer an automated method for calculating the polymer glass transition temperature with `glass_transition.py`, which automatically begins the simulation from the trajectory of the final annealing cycle from `equilibrate.py`. It can be run using a similar submission script to `submit.slm`. Full details on the glass transition calculation can be found in the preprint link: https://arxiv.org/abs/2603.05362.

## Polymer database

Data collected on 103 homopolymers using PolyRapid can be found in `results.csv`. Where available, experimental results for the polymer density and glass transition temperatures have also been provided. Output configuration files for all equilibrated polymers, along with their topology files, can be found in `output_boxes/`.

