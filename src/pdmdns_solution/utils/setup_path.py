import sys
from pathlib import Path


necotis_solution_directory = Path(__file__).parent.parent.absolute()
root_dir = necotis_solution_directory.parent
intel_code_directory = root_dir / "intel_code"
microsoft_code_directory = root_dir / "microsoft_dns"


sys.path += [
    str(x)
    for x in [
        root_dir,
        necotis_solution_directory,
        intel_code_directory,
        microsoft_code_directory,
    ]
]
