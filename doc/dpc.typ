#let title = "DPC for IHM"
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

= NMPC formulation

== IHM1
$
  min space    & space sum_(k=0)^(N_f-1) q_"XY" (X_k-X_k^"ref")^2 + q_"XY" (Y_k-Y_k^"ref")^2 + q_phi (phi_k-phi_k^"ref")^2 + q_v (v_k-v_k^"ref")^2 + q_delta delta_k^2 + q_T T_k^2 \
  "s.t." space & space x_(k+1) = f(x_k, u_k), space k = 0, ..., N_f-1 \
               & space -T_max <= T_k <= T_max, space k=0, ..., N_f -1\
               & space -delta_max <= delta_k <= delta_max, space k=0, ..., N_f -1
$

where $x=(X,Y,phi,v)^T$ denotes the state of the system, $u=(T,delta)^T$ its
control input, and $f$ the discretized dynamics coming from the following ODE:

$
  dot(X) = v cos(phi + beta) \
  dot(Y) = v sin(phi + beta) \
  dot(phi) = v sin(beta) / L \
  dot(v) = F_x/m \
$
where $beta = 1/2 delta$ denotes the kinematic slip angle, $L$ the wheelbase, $m$ the
mass of the car, and $F_x = C_m T - C_(r 0) - C_(r 1) v - C_(r 2) v^2$ the
longitudinal force applied to the car.

== IHM1.5

We only replace the costs on the XY by rotating the error accordingly to obtain
longitudinal and lateral errors:
$
  e_("lon",k) & = cos(phi_k^"ref") (X_k - X_k^"ref") + sin(phi_k^"ref") (Y_k - Y_k^"ref") \
  e_("lat",k) & = -sin(phi_k^"ref") (X_k - X_k^"ref") + cos(phi_k^"ref") (Y_k - Y_k^"ref")
$
and then adding the cost $q_"lon" e_("lon",k)^2 + q_"lat" e_("lat",k)^2$ #h(0.1mm) .

Then only the weight matrix changes, but the cost function remains a linear
least-squere loss. In particular, the weight matrix is different at each stage
(because it depends on $phi^"ref"$).

We can also use these newly defined lateral error to enforce (approximate) track constraints as 
$
  e_("lat,min,k") <= e_("lat", k) <= e_("lat,max,k")
$
so in the end the OCP reads
$
  min space    & space sum_(k=0)^(N_f-1) q_"lon" e_("lon", k)^2 + q_"lat" e_("lat", k)^2 + q_phi (phi_k-phi_k^"ref")^2 + q_v (v_k-v_k^"ref")^2 + q_delta delta_k^2 + q_T T_k^2 \
  "s.t." space & space x_(k+1) = f(x_k, u_k), space k = 0, ..., N_f-1, \
               & space e_("lat,min,k") <= e_("lat", k) <= e_("lat,max,k"), space k = 0, ..., N_f, \
               & space -T_max <= T_k <= T_max, space k=0, ..., N_f -1, \
               & space -delta_max <= delta_k <= delta_max, space k=0, ..., N_f -1 space .\
$


== DPC

inputs: current state and state reference
