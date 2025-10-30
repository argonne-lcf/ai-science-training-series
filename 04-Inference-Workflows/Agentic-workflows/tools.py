from langchain_core.tools import tool


@tool
def calculator(expression: str) -> str:
    """Evaluate mathematical expressions safely.

    This function provides a safe way to evaluate mathematical expressions
    using numexpr. It supports basic mathematical operations and common
    mathematical functions.

    Parameters
    ----------
    expression : str
        Mathematical expression to evaluate (e.g., "2 * pi + 5")

    Returns
    -------
    str
        String result or error message

    Notes
    -----
    Supported mathematical functions:
    - Basic operations: +, -, *, /, **
    - Trigonometric: sin, cos, tan
    - Other: sqrt, abs
    - Constants: pi, e
    """
    import math
    import numexpr

    local_dict = {
        "pi": math.pi,
        "e": math.e,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "sqrt": math.sqrt,
        "abs": abs,
    }

    try:
        cleaned_expression = expression.strip()
        if not cleaned_expression:
            return "Error: Empty expression"

        result = numexpr.evaluate(
            cleaned_expression,
            global_dict={},
            local_dict=local_dict,
        )

        if isinstance(result, (int, float)):
            return f"{float(result):.6f}".rstrip("0").rstrip(".")
        return str(result)

    except Exception as e:
        return f"Error evaluating expression: {e!s}"


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
