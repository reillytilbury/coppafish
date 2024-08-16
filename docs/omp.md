# Orthogonal Matching Pursuit (OMP)

OMP is coppafish's current best gene assignment algorithm. OMP runs independently, except requiring 
[call spots](#call-spots) for more accurate representation of each gene's unique barcode: its bled code 
($\mathbs{b}_{grc}$). Also, OMP does not have any understanding about the difference between rounds and channels.

## Maths Definitions

- $r$ and $c$ represents sequencing rounds and channels respectively.
- $\mathbf{B}_{grc}$ represents gene g's bled code in round $r$, channel $c$.
- $\mathbf{S}_{prc}$ is pixel $p$'s colour in round $r$, channel $c$, after pre-processing is applied.
- $\mathbf{c}_{pgi}$ is the OMP coefficient given to gene $g$ for image pixel $p$ on the $i$'th iteration. $i$ takes 
values $1, 2, 3, ...$

## 0: Pre-processing

All pixel colours are gathered using the results from register. Any out of bounds round/channel colour intensities are 
set to zero. Pixel colours are normalised based on their intensities. If $\widetilde{\mathbf{S}}$ are the initial 
pixel colours, the final pixel colours $\mathbf{S}$ become

$$
\mathbf{S}_{prc} = \frac{\widetilde{\mathbf{S}}_{prc}}{\sqrt{\sum_{rc} \widetilde{\mathbf{S}}_{prc}}}
$$

## 1: Assigning the Next Gene

A pixel can have more than one gene assigned to it. Let's say we are on iteration $i$ for pixel $p$. The pixel will 
already have $i - 1$ genes assigned to it and their coefficients have already been computed ($\mathbf{c}_{pg(i - 1)}$). 
We compute the latest residual pixel colour $\mathbf{R}_{prci}$ as 

$$
\mathbf{R}_{prci} = \mathbf{S}_{prc} - \sum_g \mathbf{c}_{pg(i - 1)}\mathbf{B}_{grc}
$$

For the first iteration, $\mathbf{R}_{prc(i=1)} = \mathbf{S}_{prc}$. Using this residual, a weighted dot product score 
is computed for every gene $g$ as 

$$

$$

## 2: Gene Coefficients

On each iteration, the gene coefficients are computed for the genes assigned to pixel $p$ to best represent the 
pixel's colour. All unassigned genes have a zero coefficient. The coefficients vector of length $g$, 
$\mathbf{c}_{pgi}$ are computed through the method of least squares by minimising the scalar residual 

$$
\sum_{rc}(\mathbf{S}_{prc} - \sum{g}\mathbf{B}_{grc}\mathbf{c}_{pgi})^2
$$

In other words, using matrix multiplication, the coefficient vector of length genes is 

$$
\mathbf{c} = \bar{\mathbf{B}}^{-1} \bar{\mathbf{S}}
$$

where $\bar{...}$ represents flattening the round and channel dimensions into a single dimension, so 
$\bar{\mathbf{B}}$ is of shape $\text{genes}$ by $\text{rounds} * \text{channels}$ and $\bar{\mathbf{S}}$ is of shape 
$\text{rounds} * \text{channels}$. $(...)^{-1}$ is the Moore-Penrose matrix inverse (a psuedoinverse).

