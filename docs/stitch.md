# Stitch

The stitch section is the part of the pipeline responsible for creating a global coordinate system. This is done by looking at the overlapping regions between tiles and seeing how they have deviated from their expected positions.

<p align="center">
<img src="https://github.com/user-attachments/assets/2a0a4c18-65a8-4276-bd61-3e03f351c686" width="600">
<br />
<span>A Stitched collection of 6 tiles. </span>
</p>

## Algorithm
The origin of each tile $t_i$, which we call $\mathbf{X}_i$ is the position of its top left corner in the global coordinate system. Each tile $t_i$ is given a _nominal origin_ $\mathbf{\tilde{X}}_i$ given by

$$ 
\mathbf{\tilde{X}}_i = T(1-r) \bigg( Y_i, X_i, 0 \bigg),
$$

where 

- $T$ is the size in pixels of the tile. We typically have $T = 2304$,

- $r \in (0, 1)$ is the expected overlap between tiles and is chosen on the microscope,

- $Y_i$ and $X_i$ are the integer indices of the tile position in y and x respectively,

- we are using our default system of writing vectors in $yxz$ format.

These origins are corrected by shifts $\mathbf{S}_i$ which capture small deviations from the nominal origins, ie:  $\mathbf{X}_i = \mathbf{\tilde{X}}_i + \mathbf{S}_i$. These shifts are found as follows:

<ol>
  <li>
    Compute shift $\mathbf{v_{ij}}$ between the overlapping region of all adjacent tiles $t_i$ and $t_j$ using <a href="https://en.wikipedia.org/wiki/Phase_correlation#:~:text=Phase%20correlation%20is%20an%20approach,calculated%20by%20fast%20Fourier%20transforms.">phase correlation</a>. This would be $\mathbf{0}$ if there were no deviations from the expected origins.
  </li>

  <li>
    Assign each shift $\mathbf{v}_{ij}$ a score $\lambda_{ij}$. We use 

    $$
    \lambda_{ij} = \mathrm{corr}_ {\mathbf{x}}(t_i(\mathbf{x - v_{ij}}), t_j(\mathbf{x}))^2.
    $$
  </li>

  <li>
    We typically have about twice as many independent shifts as tiles (one south and one east for each non-boundary tile). This means our problem is over-constrained and doesn't have an exact solution. We can get an approximate solution for each shift $\mathbf{S_i}$ by minimizing the loss function

    $$
    L(\mathbf{S}) = \sum_{i, j \ \mathrm{neighb}} \lambda_{ij} |\mathbf{S}_i - \mathbf{S}_j - \mathbf{v_{ij}}|^2,
    $$

    which is just saying that the deviations from the nominal origins for tiles $i$ and $j$ should be close to the observed deviations $\mathbf{v_{ij}}$, and moreover that we should care more about making these similar when the deviations $\mathbf{v_{ij}}$ are high quality, i.e., $\lambda_{ij}$ is large.
  </li>

  <li>
    Differentiating the quadratic equation above gives a linear equation $\mathbf{AS} = \mathbf{B}$. However, $\mathbf{A}$ is not invertible, so this does not have a unique solution (any common translation of all $\mathbf{S}_i$ is another solution). We therefore solve for $\mathbf{S}$ by minimizing a second loss function

    $$
    J(\mathbf{S}) = ||\mathbf{AS} - \mathbf{B}||^2,
    $$
    which just says that we take the smallest of all the shifted solutions for $\mathbf{S}$. This could have also been achieved by adding a regularization term $\beta ||\mathbf{S}||^2$ to the original loss function $L$.
  </li>
</ol>

## Image Fusing
Once we have the final origins $\mathbf{X}_i$, we can fuse the images together. Overlapping regions are linearly tapered to avoid sharp transitions.

<p align="center">
<img src="https://github.com/user-attachments/assets/cfafd743-401c-4a3b-90c0-6a478ef76275" width="600">
<br />
<span>Tapered edges can be seen clearly when some tiles finish in $z$ before others. </span>
</p>


## Global Coordinates and Duplicate Spots
Each pixel $p$ has local coordinate $\mathbf{q}_p$ which we can convert to global coordinates $\mathbf{Q}_p$  by adding the origin of pixel $p$'s parent tile, ie:  $\mathbf{Q}_p = \mathbf{q}_p + \mathbf{X}_{t(p)}$. 

As discussed above, this is useful for viewing the spots in a global coordinate frame. It also allows us to remove duplicate spots at the overlap between tiles. This reduces computation time and ensures that no genes are double counted, which would skew the results of downstream analysis.

When performing gene assignments, we only use the pixels $p$ on tile $t$ whose closest tile centre in global coordinates is tile $t$. This takes care of duplicate spots in a way which doesn't actually have to view 2 spots overlapping in global coordinates, which would be error-prone.

<p align="center">
<img src="https://github.com/user-attachments/assets/aad9076f-1fe1-45a8-a2ba-d280ffcc766f" width="600">
<br />
<span>We only keep the spots on tile $t$ who are closer to the centre of $t$ than any other tile. </span>
</p>

## Diagnostics

### View Stitch Checkerboard

```pseudo
from coppafish.plot.stitch import view_stitch_checkerboard
view_stitch_checkerboard(nb)
```

This function plots tiles in an alternating green and red checkerboard pattern with overlapping regions in yellow.

<p align="center">
<img src="https://github.com/user-attachments/assets/c3eca44c-cd58-48cf-a13f-9e70123eafcb" width="600">
<br />
</p>
