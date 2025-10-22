from parsl_config import polaris_config
import parsl
from parsl.app.app import python_app

@python_app
def hello():
    return "Hello"

if __name__ == "__main__":
    with parsl.load(polaris_config):
        future = hello() #  Run water as a demonstration (O is the SMILES for water)
        print("The python app returns a future object:", future, flush=True)

        out = future.result()
        print(f"The result of the future is {out}", flush=True)
