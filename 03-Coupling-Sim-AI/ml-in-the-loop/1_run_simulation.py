from chemfunctions import compute_vertical
from parsl_config import polaris_config
import parsl
from parsl.app.app import python_app

compute_vertical_app = python_app(compute_vertical)

if __name__ == "__main__":
    with parsl.load(polaris_config):
        future = compute_vertical_app('O') #  Run water as a demonstration (O is the SMILES for water)
        print("The python app returns a future object:", future, flush=True)

        ie = future.result()
        print(f"The ionization energy of {future.task_record['args'][0]} is {ie:.2f} eV", flush=True)
