"""
The purpose of this Python script is to edit the yml file in the hope
that MaTPIP environment installation succeeds. To be more precise, the
aforementioned yml file specifying the MaTPIP virtual environment
consists of two sections, the first of which installs packages using
Conda and the second of which does so using pip. Regarding the first
section, each and every package has two additional specifications,
namely the desired version as well as the desired build. It is
hypothesised that removing the build specification could make the
environment installation via the yml file work as the build is
presumably hardware-specific.
"""

# The `VACV_screen` directory is the parent directory, which is why even
# though this script is located in the same subdirectory as the yml
# file, the yml file's path has to be specified relative to parent
# directory
path_to_yml_file = "MaTPIP/py381_gpu_param.yml"

# Bear in mind that the `with` context manager is preferred in the
# context of working with files as it automatically takes care of
# closing files, even in case of an error/exception
with open(path_to_yml_file, "r") as f:
    yml_file_lines = f.readlines()

# Now, iterate over the individual lines of the yml file and remove the
# build specification, if present
for i, line in enumerate(yml_file_lines):
    if line == "  - pip:\n":
        break

    # Converting strings into lists of individual characters as
    # intermediate facilitates string analysis and manipulation
    line_list = list(line)
    if line_list.count("=") == 2:
        equality_sign_indices = [
            i for i, char in enumerate(line_list) if char == "="
        ]
        
        # Remove everything from the the second equality sign onwards is
        # removed
        yml_file_lines[i] = line[:equality_sign_indices[1]] + "\n"

# Save the lines without build specifications to a new yml file
with open("MaTPIP/py381_gpu_param_without_build_specs.yml", "w") as f:
    f.writelines(yml_file_lines)