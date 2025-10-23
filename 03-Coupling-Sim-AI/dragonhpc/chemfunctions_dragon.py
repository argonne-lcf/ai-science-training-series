"""Functions used to perform chemistry tasks in the DragonHPC workflow 
"""

from io import StringIO

import numpy as np
from ase.io import read
from ase.optimize import LBFGSLineSearch
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from xtb.ase.calculator import XTB
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from rdkit.Chem import rdFingerprintGenerator

from dragon.data.ddict import DDict


"""SIMULATION FUNCTIONS: Quantum chemistry parts of the workflow"""
def generate_initial_xyz(mol_string: str) -> str:
    """Generate the XYZ coordinates for a molecule.
    
    Args:
        mol_string: SMILES string

    Returns:
        - InChI string for the molecule
        - XYZ coordinates for the molecule
    """

    # Generate 3D coordinates for the molecule
    mol = Chem.MolFromSmiles(mol_string)
    if mol is None:
        raise ValueError(f'Parse failure for {mol_string}')
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=1)
    AllChem.MMFFOptimizeMolecule(mol)

    # Save geometry as 3D coordinates
    xyz = f"{mol.GetNumAtoms()}\n"
    xyz += mol_string + "\n"
    conf = mol.GetConformer()
    for i, a in enumerate(mol.GetAtoms()):
        s = a.GetSymbol()
        c = conf.GetAtomPosition(i)
        xyz += f"{s} {c[0]} {c[1]} {c[2]}\n"

    return xyz

def compute_vertical_dragon(dd: DDict) -> None:
    """Run the ionization potential computation

    Args:
        dd: Dragon distributed dictionary
    """
    smiles_list = dd['smiles']
    ie = []

    for smiles in smiles_list:
        # Make the initial geometry
        xyz = generate_initial_xyz(smiles)
        
        # Make the XTB calculator
        calc = XTB(accuracy=0.05)
        
        # Parse the molecule
        atoms = read(StringIO(xyz), format='xyz')

        # Compute the neutral geometry
        #  Uses QCEngine (https://github.com/MolSSI/QCEngine) to handle interfaces to XTB
        atoms.calc = calc
        dyn = LBFGSLineSearch(atoms, logfile=None)
        dyn.run(fmax=0.02, steps=250)
        
        neutral_energy = atoms.get_potential_energy()

        # Compute the energy of the relaxed geometry in charged form
        charges = np.ones((len(atoms),)) * (1 / len(atoms))
        atoms.set_initial_charges(charges)
        charged_energy = atoms.get_potential_energy()
        ie.append(charged_energy - neutral_energy)

    dd['ie'] = ie