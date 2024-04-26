#set document(title: "Derivation of the kinematic bicycle model")
#set page(margin: 1cm)

#figure(image("fig.svg", width: 100%), caption: [
  Summary of the kinematic bicycle model. \
  We ommited the $z$ axis for simplicity.
])
// #let ddt()
#let partial = sym.partial
#let diff = sym.diff

= Derivation of transport formula
Consider a rotating refrence frame ${arrow(x), arrow(y), arrow(z)}$ with an
angular velocity $arrow(omega)$ with respect to the inertial frame ${arrow(x)_i, arrow(y)_i, arrow(z)_i}$.
By Poisson formula, we have $(partial arrow(x)) / (diff t) = arrow(omega) times arrow(x) $.

Further consider an arbitrary time-dependent vector $arrow(u)(t)$ that we can
expressed as $arrow(u) = u_x arrow(x) + u_y arrow(y) + u_z arrow(z)$. If we
apply the Poisson formula to $arrow(u)$ we get:
$
  (partial arrow(u)) / (partial t) & = (partial (u_x arrow(x))) / (partial t) + (partial (u_y arrow(y))) / (partial t) + (partial (u_z arrow(z))) / (partial t) \
                                   & = (partial u_x) / (partial t) arrow(x)_b + (partial u_y) / (partial t) arrow(y)_b + (partial u_y) / (partial t) arrow(z)_b + u_z (partial arrow(z)) / (partial t) + u_y (partial arrow(y)) / (partial t) + u_z (partial arrow(z)) / (partial t) \
                                   & = (partial u_x) / (partial t) arrow(x)_b + (partial u_y) / (partial t) arrow(y)_b + (partial u_y) / (partial t) arrow(z)_b
  + arrow(omega) times arrow(u)
$

= Derivation of the yaw rate of $arrow(T)$ solely based on differential geometry

The path taken by the kinematic bicycle is given by $(X(t), Y(t))$.

$ beta = arctan(l_R/(l_R+l_F) tan(delta)), space R=sin(beta)/l_R $

= derive curvature of path taken by a kinematic bicycle
= derive $dot(v)_x,dot(v)_y$ using transport formula
= on body frame derive $dot(v)$ from that
= second derivation of $dot(v)$ directly by applying transport formula to T,N, which also gives us the lateral acceleration
$  $

= Car yaw rate
$ phi + beta = op("arctan2")(dot(Y), dot(X)) => r = dot(phi) = v kappa - dot(beta)$

= Position dynamics
$
  dot(X) & = v_x cos(phi) - v_y sin(phi) = v cos(beta) cos(phi) - v sin(beta) sin(phi) = v cos(phi + beta) \
  dot(Y) & = v_x sin(phi) + v_y cos(phi) = v cos(beta) sin(phi) + v sin(beta) cos(phi) = v sin(phi + beta) \
$