import nbformat

notebook_path = 'linguistic-agnostic-ser.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

for cell in nb.cells:
    if cell.cell_type == 'code' and "EXP8 = 8" in cell.source and "files_8 = list(input_dir_8.glob" in cell.source:
        # We replace the buggy determine SNR block
        old_snr_block = """        # Determine SNR
        if "clean" in parts:
            snr = "clean"
        else:
            for p in parts:
                if p.startswith("SNR"):
                    snr = p.replace("SNR", "")
                    break"""
        
        new_snr_block = """        # Determine SNR
        snr = "clean" # Default to clean if no SNR tag is found
        for p in parts:
            if p.startswith("SNR"):
                snr = p.replace("SNR", "")
                break"""
        
        cell.source = cell.source.replace(old_snr_block, new_snr_block)

with open(notebook_path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print("Fixed default clean detection in EXP10 aggregation cell.")
