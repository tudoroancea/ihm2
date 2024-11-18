#let title = "Motion planning"
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

= Closed tracks

In this section we consider the problem of constructing a cubic spline to
describe a 2D trajectory based on a sequence of $N$ points ${(X_i, Y_i)}_(i=0,dots,N-1)$.
Here we choose by convention that the last and first point are not the same.

== Spline fitting

We will create two functions $X(s)$ and $Y(s)$ of the track progress $s$ as 1D
cubic splines, i.e. $cal(C)^2$ functions that are piecewise cubic polynomials.// together with an approximation of the track progress associated to these points ${s_i}_(i=1,dots,N)$.
We approximate the path progress between two consecutive point $(X_i,Y_i)$ and $(X_(i+1), Y_(i+1))$ by
the euclidean distance between two consecutive points:
$
  Delta s_i = sqrt((X_(i+1)-X_i)^2 + (Y_(i+1)-Y_i)^2) " for " i=0,dots,N-2 " and " Delta s_(N-1) = sqrt((X_0-X_(N-1))^2 + (Y_0-Y_(N-1))^2),
$
so that we can define the global path progress for each point as
$
  s_0=0 " and " s_j = sum_(j=0)^(i-1) Delta s_j " for " i=1,dots,N.
$
Note that the formula above includes the definition of the total path length $s_N$.

We will define the spline intervals as $[s_i, s_(i+1)]$ for $i=1,dots,N-2$ and $[s_(N-1), s_0]$ (to
define the $N$-th interval closing the spline).

Then we write $X(s)$ and $Y(s)$ as
$
  X(s) = phi^X ((s-s_i)/(Delta s_i)),Y(s) = phi^Y ((s-s_i)/(Delta s_i)) " for " s_i <= s <= s_(i+1) " and " i=0,dots,N,
$
where the functions $phi^X_i$ and $phi^Y_i$ are the cubic polynomials on each
interval:
$
  phi^(X\/Y)_i (r) = a^(X\/Y)_i + b^(X\/Y)_i r + c^(X\/Y)_i r^2 + d^(X\/Y)_i r^3 " for " 0 <= r <= 1
$

To enforce the smoothness constraints at the junctions between the intervals, we
impose the following constraints (we write them for $X(s)$ only for simplicity
but the same translates to $Y(s)$):

- $X(s_i^-) = X(s_i^+)$ which translates to
  $
    phi_(i-1)^X ((s_i - s_(i-1))/(Delta s_(i-1))) = phi^X_i ((s_i-s_i)/(Delta s_i)) & <==> phi_(i-1)^X (1)= phi^X_i (0) \
                                                                                    & <==> a^X_(i-1)+b^X_(i-1)+c^X_(i-1) + d^X_(i-1) = a^X_i \
                                                                                    & <==> a^X_(i-1)+b^X_(i-1)+c^X_(i-1) + d^X_(i-1) - a^X_i = 0
  $
- $X'(s_i^-) = X'(s_i^+)$ which translates to

  $
    phi_(i-1)^X ' ((s_i - s_(i-1))/(Delta s_(i-1))) 1/(Delta s_(i-1)) = phi^X_i ' ((s_i-s_i)/(Delta s_i))1/(Delta s_(i-1)) & <==> (b^X_(i-1)+2 c^X_(i-1)+ 3 d^X_(i-1)) / (Delta s_(i-1)) = b^X_i / (Delta s_i) \

                                                                                                                           & <==> b^X_(i-1)+2 c^X_(i-1)+ 3 d^X_(i-1) - rho_(i-1) b^X_i = 0
  $

  where we defined $rho_i = (Delta s_(i-1)) / (Delta s_i)$.

- $X''(s_i^-) = X''(s_i^+)$ which translates to

  $
    phi_(i-1)^X '' ((s_i - s_(i-1))/(Delta s_(i-1))) 1/(Delta s_(i-1))^2 = phi^X_i '' ((s_i-s_i)/(Delta s_i))1/(Delta s_(i-1))^2 & <==> (2 c^X_(i-1)+ 6 d^X_(i-1)) / (Delta s_(i-1))^2 = (2 c^X_i) / (Delta s_i)^2 \

                                                                                                                                 & <==> 2 c^X_(i-1) + 6 d^X_(i-1) - 2 rho_(i-1)^2 c^X_i = 0
  $

If we concatenate all the coefficients $a^X_i, b^X_i, c^X_i, d^X_i$ in a single
vector $p$ we can write the following linear system of equations:

#set math.mat(gap: 0.7em)

$
  mat(
    1, 1, 1, 1, -1;, 1, 2, 3, , -rho_0;, , 2, 6, , , -rho_0^2;, , , , 1, 1, 1, 1, -1;, , , , , 1, 2, 3, , -rho_0;, , , , , , 2, 6, , , -rho_0^2;, , , , , , , , dots.down, , , , dots.down;, , , , , , , , , dots.down;
  ) p = vec(0, 0, 0, 0, 0, 0)
$

#grid(
  columns: 24,
  
)[1,2,4]

== Curvature minimization

// First version

= Open tracks

