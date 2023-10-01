# IHM2

This repository showcases a summary of the skills and knowledge I acquired 
between 2021 and 2022 at the [EPFL Racing Team](https://epflracingteam.ch/en/).
This student led initiative designs and builds each year a new electric autonomous 
racecar taking part in the [Formula Student competitions](https://www.imeche.org/events/formula-student).
I was in charge of the control of the autonomous racing car *Ariane*.

It features:
- A Nonlinear Model Predictive Controller (NMPC) based on a path-parametric bicycle 
  model with nonlinear path and control input constraints.
- A simple simulator based on a bicycle model as well for Model-in-the-Loop (MiL) 
  testing.
- A ROS 2 workspace using [Foxglove Studio](https://foxglove.dev/studio/) for 
  visualization and debugging, with its dependencies managed through 
  [miniforge](https://github.com/conda-forge/miniforge) and 
  [robostack](https://robostack.github.io/index.html)
  for cross-platform ROS compatibility.
  
> **No part of this code** (except for some external dependencies) was used at 
> any point by the EPFL Racing Team. Everything was entirely developed
> after my departure for the sole purpose to display my skills and knowledge in vehicle modelling,
> control and software development.

## Workspace setup

1. clone this repository:
  ```shell
  git clone https://github.com/tudoroancea/ihm2 --filter=blob:none --recurse-submodules
  cd ihm2
  ```
2. install miniforge3:
  ```shell
  curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
  bash Miniforge3-$(uname)-$(uname -m).sh
  ```
3. create a conda env
  ```shell
  mamba env create -f env.yml
  ```
4. install acados
  ```shell
  mamba activate ihm2
  cd $HOME/miniforge3/envs/ihm2/src
  git clone https://github.com/acados/acados.git --recurse-submodules --filter=blob:none
  cd acados
  mkdir build && cd build
  cmake -DACADOS_WITH_OPENMP=ON ..
  make install -j6
  echo "export ACADOS_SOURCE_DIR=\$HOME/miniforge3/envs/ihm2/src/acados" >> ~/miniforge3/envs/ihm2/setup.sh
  echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:\$ACADOS_SOURCE_DIR/lib" >> ~/miniforge3/envs/ihm2/setup.sh
  echo "export DYLD_LIBRARY_PATH=\$DYLD_LIBRARY_PATH:\$ACADOS_SOURCE_DIR/lib" >> ~/miniforge3/envs/ihm2/setup.sh
  mamba deactivate
  mamba activate ihm2
  pip3 install -e $ACADOS_SOURCE_DIR/interfaces/acados_template
  ```
5. build workspace (first `cd` back to the dir where you cloned this repo):
  ```shell
  chmod +x scripts/*.sh
  ./scripts/build.sh
  ```

### Foxglove Studio
 
You can download the latest version of Foxglove Studio from
[here](https://foxglove.dev/studio). To visualize data, you can then import the 
created layout in [`templates/ihm2_foxglove.json`](templates/ihm2_foxglove.json) 
and run the following command to run the foxglove bridge:
```shell
ros2 launch foxglove_ros_bridge foxglove_ros_bridge_launch.xml
```

## References & credits

- Original paper describing the NMPC formulation I based mine upon: 
  [Daniel Kloeser et al. (2020). Real-Time NMPC for Racing Using a Singularity-Free Path-Parametric Model with Obstacle Avoidance.](https://doi.org/10.1016/j.ifacol.2020.12.1376)
- The [fssim](https://github.com/AMZ-Driverless/fssim) repository from AMZ from 
  which I borrowed some STL files for the simulation environment (see 
  [this release](https://github.com/tudoroancea/ihm2/releases/tag/lego-lrt4))
- The formidable [acados](https://github.com/acados/acados) project that provided 
  me the numerical solvers for the both the simulation and the control nodes
- Johann Germanier from the EPFL Racing Team who created the small Lego model of 
  our car and gracefully provided me the STL files. 