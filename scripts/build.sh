# Copyright (c) 2023. Tudor Oancea EPFL Racing Team Driverless
# check that the script is run from the root of the brains repository
if [ $(basename "$PWD") != "ihm2" ]; then
    echo "Please run this script from the root of the brains repository"
    exit 1
fi

# makes sure that the correct python interpreter is called in CMake
PYTHON_EXE=$(which python3)
echo "Using python interpreter: $PYTHON_EXE"

if [ "$1" != "--skip_gen" ]; then
    # generate the code for the acados OCP and simulator solvers
    python3 scripts/gen_sim.py
    python3 scripts/gen_track_file.py
fi

# run colcon build
PYTHONWARNINGS=ignore:::setuptools.command.install,ignore:::setuptools.command.easy_install,ignore:::pkg_resources colcon build --symlink-install --executor parallel --cmake-args -DACADOS_PATH=$HOME/Developer/acados -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DPython3_EXECUTABLE=$PYTHON_EXE

# add stuff to DYLD_LIBRARY_PATH
echo "export DYLD_LIBRARY_PATH=\$DYLD_LIBRARY_PATH:$(pwd)/src/ihm2/generated/ihm2_dyn6_sim_gen_code" >> $(pwd)/install/setup.sh
echo "export DYLD_LIBRARY_PATH=\$DYLD_LIBRARY_PATH:$(pwd)/src/ihm2/generated/ihm2_kin6_sim_gen_code" >> $(pwd)/install/setup.sh
echo "export DYLD_LIBRARY_PATH=\$DYLD_LIBRARY_PATH:$(pwd)/src/ihm2/generated/ihm2_dyn10_sim_gen_code" >> $(pwd)/install/setup.sh
echo "export DYLD_LIBRARY_PATH=\$DYLD_LIBRARY_PATH:$(pwd)/src/ihm2/generated/ihm2_dyn6_sim_gen_code" >> $(pwd)/install/setup.sh
echo "export DYLD_LIBRARY_PATH=\$DYLD_LIBRARY_PATH:$(pwd)/src/ihm2/generated/ihm2_kin6_sim_gen_code" >> $(pwd)/install/setup.sh
echo "export DYLD_LIBRARY_PATH=\$DYLD_LIBRARY_PATH:$(pwd)/src/ihm2/generated/ihm2_dyn10_sim_gen_code" >> $(pwd)/install/setup.sh

# source the setup file
. $(pwd)/install/setup.sh