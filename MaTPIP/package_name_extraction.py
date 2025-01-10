"""
The purpose of this Python script is to extract the package names listed
in the `py381_gpu_param.yml` file and to save them to a text file.
Packages installed by Conda and Pip are each listed in a separate line.
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

conda_list = []
pip_list = []

# Now, iterate over the individual lines of the yml file and process the
# package names according to whether they are installed by Conda or Pip
# Prior to that, the indices marking the respective sections are
# determined
conda_start_index = yml_file_lines.index("dependencies:\n")
pip_start_index = yml_file_lines.index("  - pip:\n")

for i, line in enumerate(yml_file_lines):
    if (i > conda_start_index) and (i < pip_start_index):
        # Regarding the Conda section, the package name is extracted in
        # two steps
        # The first step consists of splitting the string with the
        # equality sign as separator
        # The second step uses the very first element of the list
        # generated in step one and subjects the respective string to
        # yet another split; this time, the space character serves as
        # separator; the very last element of the resulting list
        # represents the package name
        first_split = line.split("=")
        second_split = first_split[0].split()
        package_name = second_split[-1]
        conda_list.append(package_name)
    elif i > pip_start_index:
        # The approach pursued for the Pip section is similar to the one
        # employed for the Conda section; the only difference is that
        # the first split employs two successive equality signs as
        # separator instead of one
        first_split = line.split("==")
        second_split = first_split[0].split()
        package_name = second_split[-1]
        pip_list.append(package_name)

with open("MaTPIP/extracted_packages.txt", "w") as f:
    f.write("Conda packages:\n")
    f.write(" ".join(conda_list) + "\n\n")
    f.write("Pip packages:\n")
    f.write(" ".join(pip_list))