import nbformat

notebook_path = 'linguistic-agnostic-ser.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

for cell in nb.cells:
    if cell.cell_type == 'code':
        source = cell.source
        source = source.replace("['RMSE', 'R2_Score']", "['RMSE']")
        source = source.replace("OVERALL MODEL STATISTICS (RMSE & R2)", "OVERALL MODEL STATISTICS (RMSE)")
        cell.source = source

with open(notebook_path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print("Fixed R2_Score references in notebook.")
