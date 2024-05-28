#let title = "Vehicle models for autonomous driving"
#let author = "Tudor Oancea"
#set document(title: title, author: author)
#set page(margin: 1cm)
#set text(font: "New Computer Modern")
#set heading(numbering: "1.")

#let partial = sym.partial
#let diff = sym.diff

#align(
  center,
)[
  #text(size: 20pt, title)

  #text(size: 14pt, author)

  #text(size: 12pt)[#datetime.today().display("[month repr:short] [year]")]
]

= Intro

This paper serves as a summary of the knowledge I have gained in the last few
years working on car models for autonomous driving. I do my best to present a
complete and self-sufficient description of these models, their derivation,
strengths and weaknesses, and whenever possible the physical intuition behind
them. I hope that this paper will be useful to anyone interested in the subject,
and I would be happy to receive any feedback or suggestions.

kinematic bicyle model, the different car models used in the `ihm2` project, be
it in the simulator or the controllers.

model that is used in the `hess_sim` simulator. We dedicate one section to each
In this document, we present the hybrid (kinematic bicyle)-(dynamic four wheel)
of these to models, in which we present the equations that define the dynamics
of the car and try to provide some insight on the physical intuition behind
them. A third section sums up the strengths and weaknesses of each model and the
way they are both currently used in the simulator.

As usual in the litterature, we simplify the model by (almost always) ommitting
the vertical motions of the car, as well as the pitch and roll dynamics.

We will denote by $X,Y$ the position of the car in an absolute and fixed
cartesian reference frame (that we will call the _world_ or _inertial_ frame),
by $phi$ the yaw angle of the car, by $v$ the absolute velocity of the car that
has components $v_x, v_y$ in the mobile reference frame attached to the car
(that we will call the _car_ or _body_ frame), and finally by $r$ the yaw rate
of the car. The car is always actuated by a global torque $T$ (which is then
divided into wheel torques by different methods evocated in section ...) and a
steering angle $delta$.

All the following models share the following fundamental assumptions:
+ the car drives on a perfect plane, and we can therefore ignore the vertical,
  pitch and roll motions.
+ the car has a four wheel drive traction system and we can independently model
  and control each wheel.

= Kinematic models <sec:kinematic_models>

#figure(image("kin.svg", width: 100%), caption: [
  Summary of the kinematic bicycle model. \
  We ommited the $z$ axis for simplicity.
]) <fig:kin>

If we only consider the 4DOF state $x=(X, Y, phi, v)^T$ with the absolute
velocity $v$, then the dynamics are:
$
  dot(X)   & = v cos(phi+beta), \
  dot(Y)   & = v sin(phi+beta), \
  dot(phi) & = v sin(beta) / l_R, \
  dot(v)   & = 1/m (F_(x R) cos(beta) + F_(x F) cos(delta - beta)).
$ <eq:kin4>
with $beta=arctan(l_R/(l_R+l_F) tan(delta))$ the kinematic slip angle.

If instead we want to consider the 6DOF state $x=(X, Y, phi, v_x, v_y, r)^T$ that
treats separately the longitudinal velocity $v_x = v cos(beta)$, the lateral
velocity $v_y = v sin(beta)$, and the yaw rate $r$, then the dynamics are:
$
  dot(X)   & = v_x cos(phi) - v_y sin(phi), \
  dot(Y)   & = v_x sin(phi) + v_y cos(phi), \
  dot(phi) & = r, \
  dot(v)_x & = dot(v) cos(beta) - dot(beta), \
  dot(v)_y & = v sin(phi+beta), \
$ <eq:kin6>

To both models we can also add the actuator dynamics of the torque command $T$ and
the steering angle $delta$. We choose to model them as first order systems, so
that their dynamics read:
$
  dot(T)     & = 1/t_T (u_T - T), \
  dot(delta) & = 1/t_delta (u_delta - delta),
$ <eq:actuators>
where for $a in {T, delta}$, $a$ is the actual value delivered by the actuator, $u_a$ is
the actuator input, and $t_a$ is the actuator time constant.

== Derivation

=== Initial intuition and assumptions

In the case of the kinematic bicycle model, the car has to be visualized as a
system of 2 points corresponding to the front and rear axles, rigidly connected
by a weightless rod. These points have masses $m l_F/l$ and $m l_R/l$ respectively,
such that the total mass and the center of gravity coincides with the ones of
the car.

Each one of these points is subject to a single longitudinal force $F_("lon",F),F_("lon",R)$ coming
from the motors.

=== Transport formula
Consider a rotating refrence frame ${arrow(x), arrow(y), arrow(z)}$ with an
angular velocity $arrow(omega)$ with respect to the inertial frame ${arrow(x)_i, arrow(y)_i, arrow(z)_i}$.
By Poisson formula, we have $(partial arrow(x)) / (diff t) = arrow(omega) times arrow(x) $.

Further consider an arbitrary time-dependent vector $arrow(u)(t)$ that we can
expressed as $arrow(u) = u_x arrow(x) + u_y arrow(y) + u_z arrow(z)$. If we
apply the Poisson formula to $arrow(u)$ we get:
$
  (partial arrow(u)) / (partial t) & = (partial (u_x arrow(x))) / (partial t) + (partial (u_y arrow(y))) / (partial t) + (partial (u_z arrow(z))) / (partial t) \
                                   & = (partial u_x) / (partial t) arrow(x) + (partial u_y) / (partial t) arrow(y) + (partial u_y) / (partial t) arrow(z) + u_z (partial arrow(z)) / (partial t) + u_y (partial arrow(y)) / (partial t) + u_z (partial arrow(z)) / (partial t) \
                                   & = (partial u_x) / (partial t) arrow(x) + (partial u_y) / (partial t) arrow(y) + (partial u_y) / (partial t) arrow(z)
  + arrow(omega) times arrow(u)
$

Note that in the previous formulae, all the vectors are to be understood as
vectors in an abstract mathematical sense (i.e. member of a 3D euclidean space).
They are not (yet)expressed in any particular reference frame.

=== Derivation of the yaw rate of $arrow(T)$ solely based on differential geometry

The path taken by the kinematic bicycle is given by $(X(t), Y(t))$.

$
  beta = arctan(l_R/(l_R+l_F) tan(delta)) approx C delta, space R=l_R/sin(beta) \
  dot(beta) = (C(1+tan^2(delta)) dot(delta))/(1+C^2tan^2(delta)) approx C dot(delta) ((1+tan^2(delta))(1-C^2tan^2(delta)) ) = C dot(delta) (1+ (1-C^2)tan^2(delta) - C^2 tan^4(delta)) \
$

=== derive curvature of path taken by a kinematic bicycle
=== derive $dot(v)_x,dot(v)_y$ using transport formula
=== on body frame derive $dot(v)$ from that
=== second derivation of $dot(v)$ directly by applying transport formula to T,N, which also gives us the lateral acceleration
By the transport formula we get:
$
  dot(v) & = dot(v)_x arrow(x) + dot(v)_y arrow(y) + dot(v)_z arrow(z) = v cos(beta) cos(phi) - v sin(beta) sin(phi) arrow(x) + v cos(beta) sin(phi) + v sin(beta) cos(phi) arrow(y) \
$

=== Car yaw rate
$ phi + beta = op("arctan2")(dot(Y), dot(X)) => r = dot(phi) = v kappa - dot(beta)$

=== Position dynamics
$
  dot(X) & = v_x cos(phi) - v_y sin(phi) = v cos(beta) cos(phi) - v sin(beta) sin(phi) = v cos(phi + beta) \
  dot(Y) & = v_x sin(phi) + v_y cos(phi) = v cos(beta) sin(phi) + v sin(beta) cos(phi) = v sin(phi + beta) \
$

= Dynamic models

The distinction between dynamic and _kinematic_ and _dynamic_ in this paper is
not the usual one made in physics, where _kinematics_ solely refer to the
despcription of the motion of a system, and _dynamics_ to the description of the
forces and moments that explain this motion. In autonomous car models, the terms _kinematic_ and _dynamic_ are
usually used to describe the degree of consideration of the various forces that
act on the car. In the case of the kinematic models described in section
@sec:kinematic_models, even if longitudinal forces appear, most of the equations
are derived from geometric considerations. In the following subsections, all the
equations will be entirely derived from Newton's laws of motion and the
empirical modelization of the forces acting on the car.

Technically, the pure kinematic considerations (in the physical sense) would be
the description of the

== Dynamic bicycle model with 4 wheels (DYN6)
== Dynamic model with 4 wheels (DYN6+)
== Dynamic model with 4 wheels and their speeds (DYN10)
