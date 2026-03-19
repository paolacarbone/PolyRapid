import networkx as nx
from rdkit import Chem
import mbuild as mb
from networkx.algorithms import isomorphism as iso
import numpy as np
from mbuild.lib.recipes.polymer import Polymer
from openbabel import openbabel, pybel
import os
import pandas as pd
from rdkit.Chem import Descriptors



class MakeChain:
    """ Makes initial polymer chain given SMILES with connection * included """

    def __init__(self, rdkit_smiles, mbuild_smiles, n, output, separation=0.15, pid=None):
        """
        Parameters 
        ----------
        rdkit_smiles : str
            SMILES string containing '*' connection points.
        mbuild_smiles : str
            Equivalent SMILES for mBuild.
        n : int
            Number of monomers per polymer chain.
        separation : float
            Distance between connection points when joining monomers.
        """
        self.rdkit_smiles = rdkit_smiles
        self.mbuild_smiles = mbuild_smiles
        self.n = n
        self.separation = separation
        self.chain = None
        self.output = output
        self.pid = pid
        

    @staticmethod
    def rdkit_to_graph(smiles):
        """
        Build a NetworkX graph from an RDKit molecule.
        '*' atoms are treated as hydrogens ('H') for matching.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid SMILES string.")
        mol = Chem.AddHs(mol)
    
        G = nx.Graph()
    
        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            symbol = atom.GetSymbol()
            # Treat wildcard * as hydrogen
            if symbol == "*":
                symbol = "H"
            G.add_node(idx, element=symbol)
    
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            G.add_edge(i, j)
    
        return G, mol

    @staticmethod
    def mbuild_to_graph(comp):
        """
        Build a NetworkX graph from an mBuild compound.
        """
        G = nx.Graph()
        particles = list(comp.particles())
    
        for i, p in enumerate(particles):
            G.add_node(i, element=p.name)
    
        for a, b in comp.bonds():
            i = particles.index(a)
            j = particles.index(b)
            G.add_edge(i, j)
    
        return G, particles
    

    @staticmethod
    def match_rdkit_mbuild(smiles, comp):
        """
        Match RDKit (with *) to mBuild molecule using graph isomorphism.
        Returns a dict mapping RDKit atom indices to mBuild atom indices.
        """
        G_rdkit, mol = MakeChain.rdkit_to_graph(smiles)
        G_mb, particles = MakeChain.mbuild_to_graph(comp)
    
        # Define node matching function
        nm = iso.categorical_node_match("element", None)
    
        matcher = iso.GraphMatcher(G_rdkit, G_mb, node_match=nm)
    
        if matcher.is_isomorphic():
            mapping = matcher.mapping
            print("Found mbuild to rdkit match")
            return mapping, mol, particles
        else:
            raise ValueError("Cannot find a graph match. Have your SMILES been parsed properly?")
    

    # Once a graph match is found we test every combination of H's possible until we find one which makes the straightest molecule
    # according to aligned monomer vectors. An open babel optimisation makes gives the final structure. 

    def build_polymer(self):
        """Builds polymer using explicit RDKit + mBuild SMILES versions."""
        comp = mb.load(self.mbuild_smiles, smiles=True)
    
        # find matching graphs
        mapping, mol, particles = self.match_rdkit_mbuild(self.rdkit_smiles, comp)
    
        # find carbon (or any atom actually change that variable name) that's connected to the * atom 
        star_atoms = [a for a in mol.GetAtoms() if a.GetSymbol() == "*"]
        connected_carbons = [a.GetNeighbors()[0] for a in star_atoms]
        rdkit_carbon_indices = [a.GetIdx() for a in connected_carbons]
    
        # find the equivalent mbuild atom index 
        mbuild_carbon_indices = [mapping[i] for i in rdkit_carbon_indices]
    
        # find any other hydrogens bonded to the heavy atom (for making the polymer)
        particles = list(comp.particles())
        bonds = list(comp.bonds())
    
        def bonded_atoms(atom):  # not sure needed 
            bonded = []
            for a, b in bonds:
                if a == atom:
                    bonded.append(b)
                elif b == atom:
                    bonded.append(a)
            return bonded
    
        hydrogens_per_carbon = {}
        for idx in mbuild_carbon_indices:
            carbon = particles[idx]
            hydrogens = [particles.index(p) for p in bonded_atoms(carbon) if p.name == "H"]
            hydrogens_per_carbon[idx] = hydrogens
    
        # scared to touch that again
        c1, c2 = mbuild_carbon_indices 
        candidate_pairs = [(h1, h2) for h1 in hydrogens_per_carbon[c1]
                                    for h2 in hydrogens_per_carbon[c2]]
    
        def angle_between(v1, v2):
            v1 /= np.linalg.norm(v1)
            v2 /= np.linalg.norm(v2)
            return np.degrees(np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0)))
    
        best_pair = None
        best_angle = 180
        for h1_idx, h2_idx in candidate_pairs:
            h1, h2 = particles[h1_idx], particles[h2_idx]
            v1 = h1.pos - particles[c1].pos  # maybe a bit naive, v1 and v2 are the vectors from the heavy atom to whatever hydrogen for connection 
            v2 = h2.pos - particles[c2].pos
            ang = angle_between(v1, -v2)
            if ang < best_angle:  # we find the closest v1 and v2 to 180 degrees 
                best_angle = ang
                best_pair = (h1_idx, h2_idx)

        # build the polymer
        chain = Polymer()
        chain.add_monomer(
            compound=comp,
            indices=list(best_pair),
            separation=self.separation,
            replace=True
        )
    
        chain.build(n=self.n, sequence='A')

        # create PID-based directory inside output
        pid_folder = os.path.join(self.output, str(self.pid)) if self.pid else self.output
        os.makedirs(pid_folder, exist_ok=True)

        output_path = os.path.join(pid_folder, "polymer.mol2")
        chain.save(output_path, overwrite=True)
    
        print(f"Polymer built with {self.n} monomers and saved to {output_path}")
        self.chain = chain
        return chain
    
    
    def optimize_polymer(self, filename="polymer.mol2"):
        """Runs UFF optimization via OpenBabel."""

        # Build full path for input and output
        pid_folder = os.path.join(self.output, str(self.pid)) if self.pid else self.output
        input_path = os.path.join(pid_folder, filename)
        output_path = os.path.join(pid_folder, "polymer_optimized.xyz")
        
        mol = next(pybel.readfile("mol2", input_path))
        ff = openbabel.OBForceField.FindForceField("UFF")
        ff.Setup(mol.OBMol)
        ff.ConjugateGradients(200000000, 1.0e-9)
        ff.GetCoordinates(mol.OBMol)
        mol.write("xyz", output_path, overwrite=True)
        print(f"Optimization successful. Saved as {output_path}")


def make_polymers(library_csv, output_folder, monomers):
    """
    Generates polymer chains from SMILES strings.

    Args:
        library_csv (str): Path to the input CSV with SMILES.
        output_folder (str): Directory to write outputs.
        max_atoms (int): Maximum allowed atoms per generated chain.
    """

    # use explicit args instead of a config/manager object
    df_smiles = pd.read_csv(library_csv)  # Load the SMILES library
    df_smiles['Atoms_Per_Unit'] = 0
    df_smiles['Number_Of_Repetitions'] = 0
    df_smiles['Total_Atoms'] = 0
    df_smiles['Molecular_Weight_Of_Chain'] = 0.0

    os.makedirs(output_folder, exist_ok=True)  # Create the output folder

    for index, row in df_smiles.iterrows():
        smiles = row['smiles_polymer'].strip()
        smiles2 = smiles
        rdkit_smiles = smiles.replace('[*]', '*').replace('H2','').replace('H3','').replace('H','')
        mbuild_smiles = smiles2.replace('[*]', '').replace('*','').replace('()','')
        molecule = Chem.MolFromSmiles(rdkit_smiles)
        if molecule is None:
            print(f"Warning: Unable to parse SMILES string '{smiles}' at index {index}")
            continue

        molecule_with_h = Chem.AddHs(molecule)
        atoms_per_unit = molecule_with_h.GetNumAtoms()
        df_smiles.at[index, 'Atoms_Per_Unit'] = atoms_per_unit  # Number of atoms in the basic repeat unit
        mol_weight = Descriptors.MolWt(molecule_with_h)
        # compute max repeat units that fit within max_atoms using integer division
        max_atoms = (atoms_per_unit * monomers) + 2 # add two for end groups
        new_length = max_atoms // (atoms_per_unit - 2) # subtract 2 for fake end groups
        if new_length <= 0:
            print(f"Skipping PID {row.get('PID','?')}: single-unit atoms ({atoms_per_unit}) exceed max_atoms ({max_atoms})")
            continue

        total_atoms = new_length * atoms_per_unit
        df_smiles.at[index, 'Dp'] = new_length  # Degree of polymerization (repeat units)
        df_smiles.at[index, 'Total_Atoms'] = total_atoms
        df_smiles.at[index, 'Mw'] = mol_weight * new_length

        # make chains with new method

       # rdkit_smiles = smiles.replace('[*]', '*').replace('H2','').replace('H3','')
       # mbuild_smiles = smiles.replace('[*]', '').replace('()','').replace('*','')
        print(mbuild_smiles)
        print(rdkit_smiles)

        poly = MakeChain(
             rdkit_smiles=rdkit_smiles,
             mbuild_smiles=mbuild_smiles,
             n=monomers,
             separation=0.15,
             output=output_folder,
             pid=row.get('PID', f"poly_{index}")
        )

        poly.build_polymer()
        poly.optimize_polymer()


make_polymers("library.csv","Output/",20)
