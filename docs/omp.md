# Orthogonal Matching Pursuit (OMP)

OMP is coppafish's current best gene assignment algorithm. OMP runs independently, except requiring 
[register](method.md#register) for image-alignment and [call spots](method.md#call-spots) for dataset-accurate 
representation of each gene's unique barcode: its bled code ($\mathbf{B}_{grc}$). OMP does not explicitly differentiate 
between sequencing rounds and channels.

## Definitions

- $r$ and $c$ represents sequencing rounds and channels respectively.
- $\mathbf{B}_{grc}$ represents gene g's bled code in round $r$, channel $c$.
- $\mathbf{S}_{prc}$ is pixel $p$'s colour in round $r$, channel $c$, after pre-processing is applied.
- $\mathbf{c}_{pgi}$ is the OMP coefficient given to gene $g$ for image pixel $p$ on the $i$'th iteration. $i$ takes 
values $1, 2, 3, ...$
- $||...||^{(...)}$ represents an L2 norm (or Frobenius norm for a matrix) over indices within the brackets.

## 0: Pre-processing

All tile pixel colours are gathered using the results from register. Any out of bounds round/channel colour intensities 
are set to zero. 

## 1: Assigning the Next Gene

A pixel can have more than one gene assigned to it. The most genes allowed on each pixel is `max_genes` 
(typically `10`). Let's say we are on iteration $i$ ($i = 1, 2, 3, ...$) for pixel $p$. The pixel will already have 
$i - 1$ genes assigned to it and their coefficients have already been computed ($\mathbf{c}_{pg(i - 1)}$). We compute 
the latest residual pixel colour $\mathbf{R}_{prci}$ as 

$$
\mathbf{R}_{prci} = \mathbf{S}_{prc} - \sum_g(\mathbf{c}_{pg(i - 1)}\mathbf{B}_{grc})
$$

For the first iteration, $\mathbf{R}_{prc(i=1)} = \mathbf{S}_{prc}$. Using this residual, a dot product score is 
computed for every gene and background gene $g$ as 

$$
(\text{gene scores})_{pgi} = \frac{\sum_{rc}(\mathbf{B}_{grc}\mathbf{R}_{prci})}{||\mathbf{R}_{prci}||^{rc} + \lambda_d}
$$

A gene is successfully assigned to a pixel when all conditions are met:

- The best gene score is above `dp_thresh` (typically 0.225).
- The best gene is not already assigned to the pixel.
- The best gene is not a background gene.
- There are fewer than $\text{max_genes} - i + 1$ genes/background genes above the `dp_thresh` score.

The reasons for each of these conditions is:

- to remove poor gene reads and dim pixels.
- to not double assign genes.
- to avoid over-fitting on high-background pixel colour.
- to stop iterating on ambiguous pixel colour.

respectively. If a pixel fails to meet one or more of these conditions, then no more genes are assigned to it. If all 
remaining pixels fail the conditions, then the iterations stop and the coefficients $\mathbf{c}$ are kept as final.

## 2: Gene Coefficients

On each iteration, the gene coefficients are computed for the genes assigned to pixel $p$ to best represent the 
pixel's colour. All unassigned genes have a zero coefficient, so $g$ here represents only the assigned genes ($i$ 
assigned genes). The coefficients vector, $\mathbf{c}_{pgi}$, is of length $g$. $\mathbf{c}_{pgi}$ is computed through 
the method of least squares by minimising the scalar residual 

$$
\sum_{rc}(\mathbf{S}_{prc} - \sum_{g}(\mathbf{B}_{grc}\mathbf{c}_{pgi}))^2
$$

In other words, using matrix multiplication, the coefficient vector of length genes assigned is 

$$
\mathbf{c} = \bar{\mathbf{B}}^{-1} \bar{\mathbf{S}}
$$

where $\bar{(...)}$ represents flattening the round and channel dimensions into a single dimension, so 
$\bar{\mathbf{B}}$ is of shape $\text{genes assigned}$ by $\text{rounds}*\text{channels}$ and $\bar{\mathbf{S}}$ is of 
shape $\text{rounds} * \text{channels}$. $(...)^{-1}$ is the Moore-Penrose matrix inverse (a pseudo-inverse).

With the new, updated coefficients, step 1 is repeated on the remaining pixels unless $i$ is equal to `max_genes`.

## 3: Coefficient Post-Processing

The final coefficients, $\mathbf{c}_{pg}$ are normalised pixel-wise by

$$
\mathbf{c}_{pg} \rightarrow \frac{\mathbf{c}_{pg}}{||\mathbf{S}_{prc}||^{rc} + \lambda_d}
$$

$\lambda_d$ should be on the order of background signal, typically $0.4$.

## 4: Mean Sign Spot Computation
