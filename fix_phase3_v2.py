import os
import glob
import subprocess

base_dir = r"C:\Users\pedro\Documentos\NOVA IMS\2º semestre- eu\MLOps\projeto\_player_rating_extracted\mlops_player_rating"
src_dir = os.path.join(base_dir, "src", "mlops_player_rating")
core_dir = os.path.join(src_dir, "core")

if not os.path.exists(core_dir):
    os.makedirs(core_dir)

files_to_move = ["utils.py", "tracking.py"]

for f in files_to_move:
    src_file = os.path.join(src_dir, f)
    dst_file = os.path.join(core_dir, f)
    if os.path.exists(src_file):
        os.rename(src_file, dst_file)
        subprocess.run(["git", "rm", "--cached", src_file, "--ignore-unmatch"], cwd=base_dir)
        subprocess.run(["git", "add", dst_file], cwd=base_dir)

# Ensure modeling is added (since it failed last time)
subprocess.run(["git", "add", os.path.join(core_dir, "modeling.py")], cwd=base_dir)

# Update imports in all python files
py_files = glob.glob(os.path.join(base_dir, "**", "*.py"), recursive=True)
for pf in py_files:
    if "venv" in pf or "fix_" in pf:
        continue
    with open(pf, "r", encoding="utf-8") as file:
        content = file.read()
    
    new_content = content.replace("from mlops_player_rating.modeling", "from mlops_player_rating.core.modeling")
    new_content = new_content.replace("import mlops_player_rating.modeling", "import mlops_player_rating.core.modeling")
    new_content = new_content.replace("from mlops_player_rating.utils", "from mlops_player_rating.core.utils")
    new_content = new_content.replace("import mlops_player_rating.utils", "import mlops_player_rating.core.utils")
    new_content = new_content.replace("from mlops_player_rating.tracking", "from mlops_player_rating.core.tracking")
    new_content = new_content.replace("import mlops_player_rating.tracking", "import mlops_player_rating.core.tracking")
    
    if new_content != content:
        with open(pf, "w", encoding="utf-8") as file:
            file.write(new_content)

