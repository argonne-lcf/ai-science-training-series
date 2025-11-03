from typing import Any, Dict, Literal
import os
from langchain_core.tools import tool

from ase.io import write as ase_write
from ase import Atoms


@tool
def molecule_name_to_smiles(name: str) -> str:
    """Convert a molecule name to SMILES format.

    Parameters
    ----------
    name : str
        The name of the molecule to convert.

    Returns
    -------
    str
        The SMILES string representation of the molecule.

    Raises
    ------
    IndexError
        If the molecule name is not found in PubChem.
    """
    import pubchempy

    return pubchempy.get_compounds(str(name), "name")[0].canonical_smiles


@tool
def smiles_to_coordinate_file(
    smiles: str,
    output_file: str = "molecule.xyz",
    randomSeed: int = 2025,
    fmt: Literal["xyz"] = "xyz",
) -> str:
    """Convert a SMILES string to a coordinate file.

    Parameters
    ----------
    smiles : str
        SMILES string representation of the molecule.
    output_file : str, optional
        Path to save the output coordinate file (currently XYZ only).
    randomSeed : int, optional
        Random seed for RDKit 3D structure generation, by default 2025.
    fmt : {"xyz"}, optional
        Output format. Only "xyz" supported for now.

    Returns
    -------
    str
        A single-line JSON string LLMs can parse, e.g.
        {"ok": true, "artifact": "coordinate_file", "format": "xyz", "path": "...", "smiles": "...", "natoms": 12}

    Raises
    ------
    ValueError
        If the SMILES string is invalid or if 3D structure generation fails.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    # Generate the molecule object
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string.")

    # Add hydrogens and optimize 3D structure
    mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol, randomSeed=randomSeed) != 0:
        raise ValueError("Failed to generate 3D coordinates.")
    if AllChem.UFFOptimizeMolecule(mol) != 0:
        raise ValueError("Failed to optimize 3D geometry.")
    # Extract atomic information
    conf = mol.GetConformer()
    numbers = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    positions = [list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())]

    # Create Atoms object
    atoms = Atoms(numbers=numbers, positions=positions)
    ase_write(
        output_file,
        atoms,
    )

    # Return dict for LLM/tool chaining
    return {
        "ok": True,
        "artifact": "coordinate_file",
        "path": os.path.abspath(output_file),
        "smiles": smiles,
        "natoms": len(numbers),
    }


@tool
def run_mace_calculation(
    input_file: str,
    mace_model_name: str = "small",
    device: Literal["cpu", "cuda"] = "cpu",
    optimize: bool = False,
    fmax: float = 0.05,
    max_steps: int = 200,
) -> Dict[str, Any]:
    """
    Run a MACE single-point energy calculation, optionally with geometry optimization.

    Parameters
    ----------
    input_file : str
        Path to structure file readable by ASE.
    mace_model_name : str
        Name/path of the MACE model to load.
    device : {"cpu", "cuda"}, optional
        Device to run MACE on.
    optimize : bool, optional
        If True, run a geometry optimization (BFGS).
    fmax : float, optional
        Convergence force threshold (eV/Å).
    max_steps : int, optional
        Maximum number of optimization steps.

    Returns
    -------
    dict
        Calculation info, final energy, and (if optimized) final positions.
    """
    import os
    from ase.io import read
    from ase.optimize import BFGS

    # You may need to adjust this depending on how MACE is installed in your env
    from mace.calculators import mace_mp

    if not os.path.isfile(input_file):
        raise ValueError(f"Input structure file '{input_file}' does not exist.")

    # normalize device
    dev = device.lower()
    if dev not in ("cpu", "cuda"):
        dev = "cpu"

    # read atoms
    try:
        atoms = read(input_file)
    except Exception as e:
        raise ValueError(f"Could not read '{input_file}' with ASE: {e}")

    # create calculator
    try:
        calc = mace_mp(model=mace_model_name, device=dev)
    except Exception as e:
        raise ValueError(f"Could not load MACE model '{mace_model_name}'. Original error: {e}")

    atoms.calc = calc

    # if no optimization, just do single point
    if not optimize:
        energy = float(atoms.get_potential_energy())
        return {
            "status": "success",
            "message": "MACE single-point energy computed.",
            "mode": "single_point",
            "input_file": input_file,
            "mace_model_name": mace_model_name,
            "device": dev,
            "single_point_energy_eV": energy,
        }

    # otherwise run a geometry optimization
    try:
        # opt = BFGS(atoms, logfile=None)
        opt = BFGS(atoms)
        opt.run(fmax=fmax, steps=max_steps)
        converged = True
    except Exception as e:
        # even if BFGS fails we might still be able to get energy
        converged = False
        raise ValueError(f"Geometry optimization failed: {e}")

    # final energy after opt
    final_energy = float(atoms.get_potential_energy())

    # return a small LLM-friendly payload
    return {
        "status": "success",
        "message": "MACE geometry optimization completed.",
        "mode": "geometry_optimization",
        "converged": converged,
        "input_file": input_file,
        "mace_model_name": mace_model_name,
        "device": dev,
        "final_energy_eV": final_energy,
        "final_positions": atoms.get_positions().tolist(),
        "final_cell": atoms.get_cell().tolist(),
        "fmax_used": fmax,
        "max_steps_used": max_steps,
    }
