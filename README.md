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

Coming soon

### Environment

Coming soon

### Foxglove Studio
 
Coming soon

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