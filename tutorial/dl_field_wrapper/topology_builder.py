import os
import shutil
import subprocess


class TopologyBuilder:
    """
    A simple wrapper to locate and invoke the DL_field executable.
    Ensures DL_field is run from its own directory so it can read the dl_f_path file.
    """

    # maps control file parameters to their lines in the control file 
    control_line_map = {
    "output": 2,
    "forcefield": 3,

    }

    allowed_control_values = {
    "forcefield": ["CHARMM", "CHARMM22_prot", "CHARMM36_prot", "CHARMM36_nucl", "CHARMM36_lipid", "CHARMM36_carb",
                   "CHARMM36_cgenff", "CHARMM19","amber","amber16_gaff","OPLSAA","OPLS2005","OPLS_AAM","OPLS_CL_P",
                   "OPLS_DES","DREIDING","PCFF","CVFF","COMPASS","TRAPPE_EH","G54A7","INORGANIC",
                   "INORGANIC_binary_oxide","INORGANIC_ternary_oxide","INORGANIC_binary_halide","INORGANIC_glass",
                   "INORGANIC_clay","INORGANIC_zeolite","INORGANIC_zeolite_HS", "MISC_FF", "multiple"],  
    "output": ["dl_poly", "gromacs", "lammps"]           
}


    def __init__(self, dl_field_path: str = None):
        """
        If dl_field_path is provided, verify it exists and is executable.
        Otherwise, look for  on the system PATH.
        
        Raises:
            FileNotFoundError: if DL_FIELD isn't found
        """
        if dl_field_path:
            # If they pointed at a directory, assume "dl_field" lives inside it
            if os.path.isdir(dl_field_path):
                candidate = os.path.join(dl_field_path, "dl_field")
            else:
                candidate = dl_field_path

            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                self.executable = candidate
            else:
                raise FileNotFoundError(
                    f"Specified DL_field not found or not executable: {candidate}"
                )
        else:
            # No custom path: search on PATH for "dl_field"
            found = shutil.which("dl_field")
            if found and os.access(found, os.X_OK):
                self.executable = found
            else:
                raise FileNotFoundError(
                    "dl_field executable was not found on your $PATH. "
                    "Please install DL_field or provide the path to its executable."
                )

        # Store the directory containing the DL_FIELD binary
        # We need this to run DL_FIELD properly (dl_f_path file)
        self.workdir = os.path.dirname(self.executable)

    def set_input_files(self, control_file: str = None, input_file: str = None, options=None) -> None:
        """
        Writes the dl_f_path file so DL_FIELD can find the control and input files.

        Parameters:
        - control_file: path to the control file
        - input_file: path to the input structure 
        - options: dict mapping control options to values 
        """

        dl_f_path_path = os.path.join(self.workdir, "dl_f_path") # check dl_f_path is where we expect
        if not os.path.isfile(dl_f_path_path):
            raise FileNotFoundError(f"Expected dl_f_path file not found at: {dl_f_path_path}")


        if control_file:
            if not os.path.isfile(control_file):
                raise FileNotFoundError(f"control file not found! Check the path you provided.")

            with open(dl_f_path_path, "r") as f:
                rel_control = os.path.relpath(control_file, start=self.workdir)
                lines = f.readlines()

                new_lines = []
                for line in lines: # replace with correct control path
                    if line.strip().startswith("control ="):
                        new_lines.append(f"control = {rel_control}\n")
                    else:
                        new_lines.append(line)

            with open(dl_f_path_path, "w") as f:
                f.writelines(new_lines)

            if input_file: # if user provides input file, automatically change control file they provided for input structure path
                rel_structure = os.path.relpath(input_file, start = self.workdir)

                with open(control_file, "r") as f:
                    control_lines = f.readlines()

                control_lines[10] = f"{rel_structure} * configuration file \n"

                with open(control_file, "w") as f:
                    f.writelines(control_lines)




        if input_file and not control_file:
            builder_dir = os.path.dirname(__file__) # control template in the same directory as this file
            default_control = os.path.join(builder_dir, "default.control")

            if not os.path.isfile(default_control):
                raise FileNotFoundError(f"default.control not found in {builder_dir}")
            
            # find the relative directory from dl_field to default file 
            rel_control = os.path.relpath(default_control, start=self.workdir)
            print("DL_FIELD will use the default control file.") 

            # modify the dl_f_path file 
            with open(dl_f_path_path, "r") as f:
                lines = f.readlines()

            new_lines = []
            for line in lines:
                if line.strip().startswith("control ="):
                    new_lines.append(f"control = {rel_control}\n")
                else:
                    new_lines.append(line)

            with open(dl_f_path_path, "w") as f:
                f.writelines(new_lines)

            # modify the default control file to include the structure file path 
            # also include user defined control file options 

            rel_structure = os.path.relpath(input_file, start = self.workdir)

            with open(default_control, "r") as f:
                control_lines = f.readlines()

            control_lines[10] = f"{rel_structure} * config file \n"

            # check that user has entered valid options 

            if options:
                for key, value in options.items():
                    allowed = self.allowed_control_values.get(key)
                    if allowed and value.lower() not in [opt.lower() for opt in allowed]:
                        print(f"Invalid value for '{key}':'{value}'")
                        print(f"Allowed options are: {', '.join(allowed)}")
                        raise SystemExit(1)


            if options:
                for key, value in options.items():
                    linenum = self.control_line_map.get(key) # get item
                    if linenum is not None and value is not None:
                        if key == "output" and value.strip().lower() == "dl_poly":
                            control_lines[linenum] = "none * no other output \n"
                        else:
                            control_lines[linenum] = f"{value} * {key}\n"

            with open(default_control, "w") as f:
                f.writelines(control_lines)
                
        # in all cases modify where output files go

        run_dir = os.getcwd()
        output_dir = os.path.join(run_dir, "output")

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        rel_output = os.path.relpath(output_dir, start=self.workdir)

        with open(dl_f_path_path, "r") as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            if line.strip().startswith("output"):
                new_lines.append(f"output  = {rel_output}/\n")
            else:
                new_lines.append(line)

        with open(dl_f_path_path, "w") as f:
            f.writelines(new_lines)




    def run(self, capture_output: bool = True, job_dir: str | None = None) -> subprocess.CompletedProcess:
        """
        Invoke DL_field, running from its own directory so 'dl_f_path' is found.

        Parameters:
        - capture_output: bool
            If True, captures stdout 

        Returns:
            subprocess.CompletedProcess

        Raises:
            CalledProcessError if DL_field returns non-zero exit code.
        """
        cmd = [self.executable]


        # Run DL_field with cwd set to the directory that holds the binary.
        result = subprocess.run(
            cmd,
            cwd=self.workdir,
            check=True,
            stdout=subprocess.PIPE if capture_output else None,
            stderr=subprocess.PIPE if capture_output else None,
            text=True
        )

        if capture_output:
            if result.stdout:
                print(result.stdout, end="")
            if result.stderr:
                print(result.stderr, end="")

            success = "Program executed successfully." in (result.stdout or "")
            
            if not success:
                stdout_trimmed = "\n".join((result.stdout or "").splitlines()[10:])
                with open("dl_field_error.log", "a") as f:
                    f.write(f"Run directory:  {job_dir}\n\n")
                    f.write("------------------ ERROR ------------------")
                    f.write(stdout_trimmed or "(no stdout)\n")

        return result

import os
import shutil

def batch_build(builder, root_dir, control_file):
    """
    Run DL_FIELD on all subdirectories inside root_dir.
    Each subdir must contain exactly one .xyz file.
    Results will be written into that same subdir.
    """

    with open("dl_field_error.log", "w") as f:
        f.write("DL_FIELD Error Log\n")
        f.write("===================\n\n")

    # walk through immediate subdirectories
    for subdir in sorted(os.listdir(root_dir)):
        full_path = os.path.join(root_dir, subdir)
        if not os.path.isdir(full_path):
            continue

        # find the .xyz file
        xyz_files = [f for f in os.listdir(full_path) if f.endswith(".xyz")]
        if len(xyz_files) == 0:
            print(f"No .xyz file in {full_path}, skipping...")
            continue
        if len(xyz_files) > 1:
            print(f"More than one .xyz in {full_path}, using first one")
        xyz_path = os.path.join(full_path, xyz_files[0])

        print(f"Running DL_FIELD for {xyz_path}")

        # patch control file with this .xyz
        builder.set_input_files(control_file=control_file, input_file=xyz_path)
        builder.run(job_dir=full_path)

        # move DL_FIELD outputs from global output/ → subdir/
        output_dir = os.path.join(os.getcwd(), "output")
        if os.path.exists(output_dir):
            for fname in os.listdir(output_dir):
                src = os.path.join(output_dir, fname)
                dst = os.path.join(full_path, fname)
                shutil.move(src, dst)
            os.rmdir(output_dir)

        print(f"✅ Finished {subdir}, results moved to {full_path}\n")
