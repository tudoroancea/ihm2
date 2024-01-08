# Copyright (c) 2023. Tudor Oancea 
# check that the script is run from the root of the ihm2 repository
if [ $(basename "$PWD") != "ihm2" ]; then
    echo "Please run this script from the root of the ihm2 repository"
    exit 1
fi

# makes sure that the correct python interpreter is called in CMake
PYTHON_EXE=$(which python3)
echo "Using python interpreter: $PYTHON_EXE"

if [ "$1" != "--skip_gen" ]; then
    # for some reason this variable is lost in subshells
    export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:$ACADOS_SOURCE_DIR/lib
    # generate the code for the acados OCP and simulator solvers
    python3 scripts/gen_sim.py
    python3 scripts/gen_mpc.py
    python3 scripts/gen_track_file.py
fi

# run colcon build
PYTHONWARNINGS=ignore:::setuptools.command.install,ignore:::setuptools.command.easy_install,ignore:::pkg_resources colcon build --symlink-install --executor parallel --cmake-args -DACADOS_PATH=$HOME/Developer/acados -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DPython3_EXECUTABLE=$PYTHON_EXE -DCMAKE_BUILD_TYPE=RelWithDebInfo

# add stuff to DYLD_LIBRARY_PATH
echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:$(pwd)/src/ihm2/generated/ihm2_dyn6_sim_gen_code" >> $(pwd)/install/setup.sh
echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:$(pwd)/src/ihm2/generated/ihm2_kin6_sim_gen_code" >> $(pwd)/install/setup.sh
echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:$(pwd)/src/ihm2/generated/ihm2_fkin6_mpc_gen_code" >> $(pwd)/install/setup.sh
echo "export DYLD_LIBRARY_PATH=\$DYLD_LIBRARY_PATH:$(pwd)/src/ihm2/generated/ihm2_dyn6_sim_gen_code" >> $(pwd)/install/setup.sh
echo "export DYLD_LIBRARY_PATH=\$DYLD_LIBRARY_PATH:$(pwd)/src/ihm2/generated/ihm2_kin6_sim_gen_code" >> $(pwd)/install/setup.sh
echo "export DYLD_LIBRARY_PATH=\$DYLD_LIBRARY_PATH:$(pwd)/src/ihm2/generated/ihm2_fkin6_mpc_gen_code" >> $(pwd)/install/setup.sh
