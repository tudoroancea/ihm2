# Copyright (c) 2023. Tudor Oancea EPFL Racing Team Driverless
# check that the script is run from the root of the brains repository
if [ $(basename "$PWD") != "ihm2" ]; then
    echo "Please run this script from the root of the brains repository"
    exit 1
fi

# makes sure that the correct python interpreter is called in CMake
PYTHON_EXE=$(which python3)
echo "Using python interpreter: $PYTHON_EXE"

# generate the code for the acados OCP and simulator solvers
python3 model.py

# run colcon build
PYTHONWARNINGS=ignore:::setuptools.command.install,ignore:::setuptools.command.easy_install,ignore:::pkg_resources colcon build --symlink-install --executor parallel --cmake-args -DACADOS_PATH=$HOME/Developer/acados -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DPython3_EXECUTABLE=$PYTHON_EXE

# add stuff to DYLD_LIBRARY_PATH
echo "export DYLD_LIBRARY_PATH=\$DYLD_LIBRARY_PATH:$(pwd)/src/ihm2/ihm2_dyn6_sim_gen_code" >> $(pwd)/install/setup.sh
echo "export DYLD_LIBRARY_PATH=\$DYLD_LIBRARY_PATH:$(pwd)/src/ihm2/ihm2_kin4_sim_gen_code" >> $(pwd)/install/setup.sh
echo "export DYLD_LIBRARY_PATH=\$DYLD_LIBRARY_PATH:$(pwd)/src/ihm2/ihm2_ocp_gen_code" >> $(pwd)/install/setup.sh