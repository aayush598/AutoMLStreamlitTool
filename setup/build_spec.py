# setup/build_spec.py

import os
import PyInstaller.__main__

# Define project parameters
app_script = "app.py"
dist_path = "dist"
build_path = "build"
spec_path = "setup"
executable_name = "AutoML_CSV_Tool"

# Optional: include folders such as outputs or .streamlit
data_dirs = [
    ("outputs", "outputs"),  # (source, dest)
    (".streamlit", ".streamlit"),
]

# Format datas for PyInstaller --add-data
add_data_args = []
for src, dest in data_dirs:
    if os.name == "nt":
        add_data_args.append(f"{src};{dest}")  # Windows format
    else:
        add_data_args.append(f"{src}:{dest}")  # Linux/Mac format

# Build the PyInstaller command
PyInstaller.__main__.run([
    app_script,
    "--name", executable_name,
    "--onefile",
    "--noconsole",           # Set to False if you want debug output
    "--clean",
    "--distpath", dist_path,
    "--workpath", build_path,
    "--specpath", spec_path,
    *["--add-data=" + d for d in add_data_args]
])
