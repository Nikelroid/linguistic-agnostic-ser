import nbformat
import re

notebook_path = 'linguistic-agnostic-ser.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

# We want to add a mirroring step at the end of the EXP8 and EXP10 cells.
# For the aggregation cell (CSV concat):
agg_append_8 = """
# Mirror to normal results directory
import shutil
normal_summary_csv = Path(f"results/EXP{EXP}/summary/csv")
normal_summary_csv.mkdir(parents=True, exist_ok=True)
shutil.copytree(output_dir, normal_summary_csv, dirs_exist_ok=True)
print(f"Mirrored CSV summaries to {normal_summary_csv}")
"""

agg_append_10 = """
# Mirror to normal results directory
import shutil
normal_summary_csv = Path(f"results/EXP{EXP10}/summary/csv")
normal_summary_csv.mkdir(parents=True, exist_ok=True)
shutil.copytree(output_dir, normal_summary_csv, dirs_exist_ok=True)
print(f"Mirrored CSV summaries to {normal_summary_csv}")
"""

# For the plotting cells:
plot_append = """
# Mirror plots and text analysis to normal results directory
import shutil
normal_summary_dir = Path(f"results/EXP{EXP}/summary")
normal_summary_dir.mkdir(parents=True, exist_ok=True)
shutil.copytree(summary_dir, normal_summary_dir, dirs_exist_ok=True)
print(f"Mirrored full summary directory to {normal_summary_dir}")
"""

for cell in nb.cells:
    if cell.cell_type == 'code':
        # Identify EXP10 aggregation cell
        if "EXP10 = 10" in cell.source and "input_dir_10 = Path" in cell.source:
            if "shutil.copytree" not in cell.source:
                cell.source += "\n" + agg_append_10
                
        # Identify EXP8 aggregation cell
        elif "EXP = 8" in cell.source and "input_dir = Path" in cell.source and "combined_df.to_csv" in cell.source:
            if "shutil.copytree" not in cell.source:
                cell.source += "\n" + agg_append_8
                
        # Identify EXP8/EXP10 plotting cells
        elif ("EXP = 8" in cell.source or "EXP = 10" in cell.source) and "output_dir = summary_dir / \"plots\"" in cell.source:
            if "shutil.copytree" not in cell.source:
                cell.source += "\n" + plot_append

with open(notebook_path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print("Successfully injected dual-export mirror logic into notebook cells.")
