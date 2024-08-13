# Registration
The register section is the part of the pipeline concerned with aligning different rounds and channels. This is crucial for decoding spot colours into genes in the Call Spots and OMP sections. The aim of this section is to find a function $g_{trc}(\mathbf{x})$ for each tile, round and channel that takes in a location $\mathbf{x}$ on the anchor image for tile $t$ and returns the corresponding location in round $r$, channel $c$ of the same tile. Since registration is done independently for each tile and we are often only working on one tile, we sometimes omit the subscript $t$ in this documentation.

Once we have these transformations $g$, we can get a $n_{\textrm{rounds}} \times n_{\textrm{channels}}$ spot colours matrix $\boldsymbol{\zeta}(\mathbf{x})$ for each location $\mathbf{x}$ in the anchor image of a given tile via

$$
\boldsymbol{\zeta}(\mathbf{x}) =
\begin{pmatrix}
f_{0, 0}(g_{0, 0}(\mathbf{x})) & \cdots & f_{0, n_c}(g_{0, n_c}(\mathbf{x})) \\
\vdots & \ddots & \vdots \\
f_{n_r, 0}(g_{n_r, 0}(\mathbf{x})) & \cdots & f_{n_r, n_c}(g_{n_r, n_c}(\mathbf{x})) \\
\end{pmatrix},
$$

where $f_{rc}(\mathbf{x})$ is the intensity of round $r$, channel $c$ at location $\mathbf{x}$. Note that even a single poorly aligned round or channel makes this matrix difficult to decode, which highlights the importance of this section.

## High Level Overview
We need to consider a few questions when building the register pipeline. Some of the most important are:

1. How large should our search space of functions for $g$ be? Are these functions independent between rounds and channels?

2. How will we actually search this space?


### 1. Choosing the Function Space
To choose the set of functions we will use to fit to our data, we need to look for all sources of misalignment. Channel misalgnments are caused by:

- The multiple camera setup, meaning channels belonging to different cameras are often slightly shifted or rotated with respect to one another.

- [Chromatic aberration](https://en.wikipedia.org/wiki/Chromatic_aberration), the variable frequency-dependent dispersal of light through the lens. This expands the images from different channels by different amounts. See the figure below.
 

So the channel-to-channel differences are composed of shifts, rotations and scalings. We model each channel transform by an [affine transform](https://en.wikipedia.org/wiki/Affine_transformation) $A_c$.

<p align="center">
  <img src="https://github.com/user-attachments/assets/7a8e510b-9b81-4c2f-a4e1-fd28e7f34b0d" width="300" />
  <br />
  <span>An example of chromatic aberration.</span>
</p>

For round to round differences, we see much less predictable variability. Misalignments arise due to:

- Tissue expansion in $z$ throughout the rounds,

- Global shifts arising from movement of the stage,

- Variable local shifts due to the microfluidics system,

- Variable local shifts due to gravity or tissue deformation. These shifts have the potential to affect regions very differently. For example, the [pyramidal layer](https://en.wikipedia.org/wiki/Pyramidal_cell) is very densely packed and seems to sink more than surrounding areas, leading to different z-shifts in its vicinity. We have also observed rips within tissue samples, which cause different sides of the tissue to move in apart from each other in opposite directions.
 
The conclusion is that affine transformations **do not** sufficiently capture the richness of round-to-round transformations. We therefore allow for completely arbitrary transformations $\mathcal{F}_r$ for each round $r$.

??? example "Affine Transform Failure Example"
    
    The figure below shows a rip in the tissue and the resulting misalignment in the DAPI channel. This is just one example of a misalignment that cannot be captured by an affine transform.
    <p align="center">
      <img src="https://github.com/user-attachments/assets/92066d55-eeca-4783-ab34-2f853c3aedae" width="600" />
      <br />
      <span>A rip in the tissue and two attempts at affine registration. </span>
    </p>


To answer the second part of question 1 - empirically, it seems we don't need to find $n_{\textrm{rounds}} \times n_{\textrm{channels}}$ independent transforms per tile, we only need $n_{\textrm{rounds}} + n_{\textrm{channels}}$. Explicitly, we model every transform as 

$$
 g_{rc}(\mathbf{x}) = A_{c}(\mathcal{F}_{r}(\mathbf{x})).
$$

### 2. Computing the Transforms
The round transforms are computed with [Optical Flow](https://en.wikipedia.org/wiki/Optical_flow), while the channel transforms are computed with [Iterative Closest Point](https://en.wikipedia.org/wiki/Iterative_closest_point). For further details see the sections below.


??? note "Note on Affine Transforms"

    When we compute the round transforms $\mathcal{F}_r$ these often include some systematic error, like a small shift of 1 pixel and slight underestimation of the z-expansion. This is due to 
    
       - downsampling of the images used to compute the optical flow transforms,
    
       - A failure to find good shifts at z-boundaries, due to poor quality images in these planes.

    To get around this, we find an affine correction $B_r$ for each round. This means our functions $g_{rc}$ can be written as 
    
    $$
    g_{rc}(\mathbf{x}) = A_{c}(B_{r}(\mathcal{F}_{r}(\mathbf{x})).
    $$ 
    
    We combine the two affine transforms $A_c$ and $B_r$ to give us the simple formula 

    $$
    g_{rc}(\mathbf{x}) = A_{rc}(\mathcal{F}_{r}(\mathbf{x})),
    $$ 

    which means that in practice our affine maps actually depend on round as well as channel.


## Optical Flow
We use optical flow to align the anchor DAPI $D_{r_{\textrm{ref}}}(\mathbf{x})$  to the round $r$ DAPI $D_r(\mathbf{x})$. The output of this algorithm is a function $\mathcal{F}_r$ which satisfies the relation 

$$
D_{r_{\textrm{ref}}}(\mathcal{F}_r(\mathbf{x})) \approx D_r(\mathbf{x}).
$$

### 1. How does it work?
Suppose we have 2 images $I$ and $J$ which we'd like to register. This means for each position $\mathbf{x}$ in image $J$ we would like to find the shift $\mathbf{s}$ satisfying 

$$
I(\mathbf{x} + \mathbf{s}) = J(\mathbf{x}).
$$

Assuming that this $\mathbf{s}$ is small and that the function $I$ is sufficiently smooth, we can approximate it by a linear function in this neighbourhood. Taylor expanding and rearranging yields:

$$
\mathbf{s} \cdot \boldsymbol{\nabla} I(\mathbf{x}) \approx J(\mathbf{x}) -  I(\mathbf{x}),
$$

which is called the flow equation, an under-determined equation due to the fact that there are 3 unknowns - each component of $\mathbf{s}$. 

There are many different methods that exist to tackle this. The one we use is called the [Lucas-Kanade](https://en.wikipedia.org/wiki/Lucas%E2%80%93Kanade_method) method, which assumes that all pixels in a small window of radius $r$ (the `window_radius` parameter with default value 8) around the point $\mathbf{x}$ have the same shift $\mathbf{s}$. 

Since this method assumes that all pixels have the same shift within this window, the condition that $I$ is smooth is very important, as we need to ensure that the same flow equation holds for all $x$ in the window. For this to be true, the Hessian $\frac{\partial ^2 I}{\partial \mathbf{x}^2}$ cannot be too large in this window.

Lucas-Kanade works as follows. Let $\mathbf{x}_1, \cdots, \mathbf{x}_n$ be all the points in this window. Then assuming these all have the same shift $\mathbf{s}$, we can gather the $n$ flow equations

$$
\begin{pmatrix} \boldsymbol{\nabla} I(\mathbf{x}_1)^T \\ \vdots \\ \boldsymbol{\nabla} I(\mathbf{x}_n)^T  \end{pmatrix} \mathbf{s} = \begin{pmatrix} J(\mathbf{x}_1) - I(\mathbf{x}_1) \\ \vdots \\ J(\mathbf{x}_n) - I(\mathbf{x}_n)  \end{pmatrix}, 
$$

which is now overdetermined! This is a better problem to have though, as the solution can be approximated by least squares. The

The above derivation makes the following assumptions

1. The shift $\mathbf{s}$ is small,

2. The images are smooth. Ie, the Hessian $\frac{\partial ^2 I}{\partial \mathbf{x}^2}$ is not too large,

3. The images $I$ and $J$ have the same intensities, so that $I(\mathbf{x} + \mathbf{s}) = J(\mathbf{x})$.

To make sure we meet the assumptions we carry out the following steps:

1. Shift size:
    - We apply an initial [Phase Correlation](https://en.wikipedia.org/wiki/Phase_correlation) to find any global shift $\mathbf{\tilde{s}}$ between $I$ and $J$, and shift $I$ by this amount: $I(\mathbf{x}) \mapsto I(\mathbf{x} - \mathbf{\tilde{s}})$. Only once this is done do we carry out optical flow. This way, optical flow only captures the deviations from the global shift $\mathbf{\tilde{s}}$.

    - We downsample the images $I$ and $J$ in $y$ and $x$ before registration, which reduces the relative size of the shift.

    - The [implementation](https://scikit-image.org/docs/stable/api/skimage.registration.html#skimage.registration.optical_flow_ilk) we use takes an iterative approach, meaning that after the initial flow field is found the algorithm runs again until some stopping criterion is met.

2. Smoothness:

    - For practical reasons (speed and shift size), we need to downsample the images by a factor in $y$ and $x$ before registering. We perform the downsampling by taking the mean within each 4 by 4 sub-block as opposed to just extracting every 4th pixel in $y$ and $x$. This increases smoothness, as shown below.
    - We smooth the images with a small Gaussian blur before registering. This needs to be done carefully because too much blurring decreases the resolution of the images and therefore the quality of the registration.

3. To ensure we have similar intensity profiles we match the means of $I$ and $J$ in similar spatial locations.


??? example "Smoothing Example"
   
    The figures below shows the effect of downsampling and blurring on the images. The Hessian determinants are shown on the right of the images.

    === "Nearest Neighbour Downsampling"

        <p align="center">
          <img src="https://github.com/user-attachments/assets/aa150fb6-169f-49b4-b51f-d4e541061214" width="600" />
          <br />
         </p>

    === "Mean Sub-Block Downsampling"

        <p align="center">
          <img src="https://github.com/user-attachments/assets/86f75389-c855-47f8-ae42-529e452ff31c" width="600" />
          <br />
         </p>

    === "Mean Sub-Block Downsampling + Gaussian"

        <p align="center">
          <img src="https://github.com/user-attachments/assets/714dbe0d-d12d-4ca2-9e0e-b7f3aeda576d" width="600" />
          <br />
         </p>


### 2. Practical Considerations

#### Speed
Speed is an issue with this algorithm, because it needs to be run independently on so many pixels. We take the following steps to optimise it:

1. As mentioned above, we downsample the images in $y$ and $x$. The amount of donwsampling is controlled by the config parameter `sample_factor_yx` which has default value 4 in both directions, meaning that the algorithm runs 16 times faster than it would without downsampling.

2. We split the downsampled images into 16 subvolumes (4 in $y$ and 4 in $x$), and run optical flow in parallel on all of these independent subvolumes. The number of cores used can be adjusted by changing the `flow_cores` parameter though if left blank this will be computed automatically.

#### Interpolation
As mentioned previously, the algorithm assumes that the images have the same intensities. This condition is certainly satisfied near cell nuclei, where similar features exist in both images. Far from nuclei though, where all we have is noise, 
the 2 images have completely independent intensities. The result of this is that our flow fields tend to only give reliable results near nuclei, as shown below.

<p align="center">
  <img src="https://github.com/user-attachments/assets/e2674274-6115-4d93-a06c-0efd1c243727" width="600" />
  <br />
  <span> The shifts found by optical flow are only reliable in a small window around cell nuclei.</span>
</p>

This is problematic, as a lot of our reads are found in between cell nuclei! We need to interpolate the values in these regions. 

##### Hard Threshold Interpolation

Suppose we have a flow field $\mathcal{F}$ that we would like to interpolate. We might go about it in the following way:

1. Choose some locations $\mathbf{x}_1, \cdots, \mathbf{x}_n$ where we know the shifts computed $\mathbf{s}_1, \cdots, \mathbf{s}_n$ are reliable,

2. Define the interpolated flow to be of the form 

$$
\mathcal{F}_{\textrm{interp}}(\mathbf{x}) = \sum_i w(\mathbf{x}, \mathbf{x}_i) \mathbf{s}_i ,
$$

where the sum is over all sample points $\mathbf{x}_1, \cdots, \mathbf{x}_n$, and the weights $w(\mathbf{x}, \mathbf{x}_i)$ have the following properties:

- $\sum_i w(\mathbf{x}, \mathbf{x}_i) = 1$ for all $\mathbf{x},$

- $w(\mathbf{x}, \mathbf{x}_i)$ is a decreasing function in $||\mathbf{x}||$ and $w(\mathbf{x}_i, \mathbf{x}_i) \approx 1.$

If these 2 properties are met then $\mathcal{F}_{\textrm{interp}}$ will be a weighted average of all the shifts $\mathbf{s}_i$, and since the weights are decreasing, the value of the $\mathcal{F}_{\textrm{interp}}$ at each interpolation point $\mathbf{x}_i$ will be strongly weighted toward $\mathbf{s}_i$.

Do such weight functions exist? Can we construct them? Yes and yes! Define the function 

$$
K(\mathbf{x}, \mathbf{y}) = \exp \Bigg( -\frac{1}{2 \sigma^2} ||\mathbf{x} - \mathbf{y}||^2 \Bigg),
$$

then we can define the weights by

$$ 
w(\mathbf{x}, \mathbf{x}_i) = \dfrac{K(\mathbf{x}, \mathbf{x}_i)}{\sum_j K(\mathbf{x}, \mathbf{x}_j)}.
$$

It is easy to see that this satisfies both the desired properties for the weights.

??? tip "How to choose $\sigma$?"
    
    In the limits: 

    - as $\sigma \to 0$ this tends to nearest neighbour interpolation, 

    - as $\sigma \to \infty$ the image takes the same value everywhere, the $\lambda$ weighted mean of the flow image.

    Another way of saying this is that as $\sigma$ grows, so does the radius of contributing pixels.

    We expect the shifts to vary more quickly in $z$ than in $xy$, so we have a different parameter for the blurring in each direction: `smooth_sigma`. This takes default values `[10, 10, 2]` ($y$, $x$ and $z$).
 

##### Extension to Soft Threshold Interpolation
The above method works well, but having a hard threshold means that some points $\mathbf{x}_1, \cdots, \mathbf{x}_n$ are used while others are completely ignored. This can lead to undersampling. A better approach is to employ a soft threshold, where we use all points $\mathbf{x}_i$ in the flow image but weight their contributions by the quality of the match at $\mathbf{x}_i$, which we will call $\lambda(\mathbf{x}_i)$.

This results in an interpolation of the form

$$
\mathcal{F}_{\textrm{interp}}(\mathbf{x}) = \sum_i \lambda(\mathbf{x}_i) w(\mathbf{x}, \mathbf{x}_i) \mathbf{s}_i ,
$$

where the sum now ranges over all points in the image, and the weight functions are given by

$$ 
w(\mathbf{x}, \mathbf{x}_i) = \dfrac{K(\mathbf{x}, \mathbf{x}_i)}{\sum_j \lambda(\mathbf{x}_j) K(\mathbf{x}, \mathbf{x}_j)}.
$$

ADD IMAGE OF TILES BEFORE AND AFTER INTERP.

??? note "Definition of the Score $\lambda$"

    We use the score
   
    $$
    \lambda(\mathbf{x}) = D_{r_{\textrm{ref}}}(\mathcal{F}_r(\mathbf{x})) D_r(\mathbf{x}),
    $$
   
    which is just the dot product of the adjusted reference image and the target image. 

    We normalise the score to have a mean of 1 for every z-plane so that the shifts don't become overly biased towards those in the bright centre. This is an issue for the $z$ shifts, which show considerable variability between z-planes.
      

## Iterative Closest Point

We now attempt to find the affine corrections to the flows found earlier. As mentioned previously, this will be an affine transform for each tile $t$, round $r$ and channel $c$. Furthermore, it will be separated into 2 transforms:

- A round transform $B_{r}$ which corrects for any errors in the flow $\mathcal{F}_r$,

- A channel transform $A_{c}$ which corrects for all sources of the channel to channel variability.

We have omitted the tile subscript, but keep in the back of your mind that these transforms vary between tiles.

### 1. How does it work
Optical flow took in 2 **images** ($I$ and $J$ ) as inputs and returned 3 images of the same size as outputs (the flow in each direction $y$, $x$ and $z$). ICP differs in that it takes in 2 **point-clouds** as input and returns an affine transform $A$ as output. 

Let $X = \begin{pmatrix} \mathbf{x}_1, \cdots,  \mathbf{x}_m \end{pmatrix}$ be the base point cloud and $Y = \begin{pmatrix} \mathbf{y}_1, \cdots,  \mathbf{y}_n \end{pmatrix}$  be the point cloud we are trying to match this to.

In our case $X$ is the set of anchor points and $Y$ is the set of points in a given round and channel. So for every point $\mathbf{y}_i$ in $Y$, we expect there to be a corresponding point $\mathbf{x}_{\beta(i)}$ in $X$ (the converse is not true). Estimating the matching $\beta$ between base and target points is the main difficulty in ICP. Some common methods to estimate this matching include:

1. We let $\mathbf{x}_{\beta(i)}$ be the closest point from $X$ to the point $\mathbf{y}_i$,

2. The same as 1. but if there is no point in $X$ within a certain radius $r$ of $\mathbf{y}_i$ we don't bother to find a match,

3. The same as 2. but we allow different radii in $y$, $x$ and $z$. This is useful if we have some prior that the misalignment affects $y$ and $x$ more than $z$, as we typically do.

4. The same as 1. but we remove outliers where the shift $\mathbf{y}_i -  \mathbf{x}_{\beta(i)}$ seems to be very different from others in its vicinity.

We use approach 3. The parameters config parameters `neighb_dist_thresh_yx` and `neighb_dist_thresh_z` refer to $r_{yx}$ and $r_z$ respectively. 

??? warning "Setting $r_z$ too low"
    
    Currently we think that ICP is correcting $y$ and $x$ more than $z$, so we have $r_{z} < r_{xy}$. If this changes in the future (for example, if optical flow is not sufficiently capturing the variable z-shifts) then increasing $r_z$ will allow ICP to have greater impact on the $z$ transforms.

Once we have a matching $\beta$, ICP works by finding an affine map $A$ minimising the loss function 

$$
L(A) = \sum_{i} || A \mathbf{x}_{\beta(i)} - \mathbf{y}_i ||^2,
$$

(where the sum is over all those elements in $Y$ that have been assigned a match) and then iterate this process of matching then minimising until some stopping criteria are met. We have the 2 following stopping criteria:
 
1. If 2 consecutive matchings are identical $\beta_{t+1} = \beta_t$ then ICP gets stuck in an infinite loop, so we stop the iterations.

2. The maximum number of iterations are reached. This is set by `icp_max_iter` which has default value 50.

The algorithm can be summarised as follows:
```pseudo
# Args:
# X = m x 4 matrix of padded base positions
# Y = n x 3 matrix of target positions
# transform_initial refers to the 4 x 3 starting estimate
# epsilon = neighb_dist_thresh
# ||a|| refers to the norm of a vector a: ||a|| = sqrt(sum_i a_i ** 2)

# initialise neighb_prev and transform
transform, neighb_prev = transform_initial, None

# Update X according to our initial guess
X = X @ transform

# begin loop
for i in range(n_iter):

    # Find closest point in X to each point in Y
    neighb = [argmin_k || X[k] - Y[j] || for j in range(n)]

    # Remove these matches if they are above the neighb_dist_thresh.
    # In the real code there are different thresholds for xy and for z
    neighb = [neighb [j] if || X[neighb[j]] - Y[j] || < epsilon, 
              else None for j in range(n)]

    # Terminate if neighb = neighb_prev for all points
    if neighb == neighb_prev:
        QUIT

    # Find transform_update minimising squared error loss
    transform_update = argmin_B sum_j ||X[neighb[i]] @ B - Y[i]|| ** 2

    # Update: 
    # - X by applying the correction transform,
    # - Our working estimate of transform,
    # - neighb_prev to the current neighb
    X = X @ transform_update 
    transform = transform_update  @ transform
    neighb_prev = neighb

```

### 2. Implementation
Let $X_{r, c}$ be the $n_{\textrm{spots}}(r,c) \times 3$ matrix of all the spots found on round $r$ channel $c$.

??? note "Min Spots Criterion"
    
    ICP will not run on a tile, round, channel with too few spots. This threshold is set by `icp_min_spots` which has default value 100.

#### Round Transform
To compute the round transforms $B_r$, we first adjust $X_{r_{\textrm{ref}}, c_{\textrm{ref}}}$ by the flow to yield $\mathcal{F}_r(X_{r_{\textrm{ref}}, c_{\textrm{ref}}})$, which should approximately put the anchor spots in round $r$ coordinates. We align these to the target points $X_{r, c_{\textrm{ref}}}$. As a formula this reads as

$$
B_r = \textrm{ICP} (\textrm{base} = \mathcal{F}_r(X_{r_{\textrm{ref}}, c_{\textrm{ref}}}), \quad \textrm{target} = X_{r, c_{\textrm{ref}}}).
$$

This should capture any systematic affine errors in the flow field $\mathcal{F}_r$.

#### Channel Transform
To compute the channel transforms $A_c$ we align the anchor points $X_{r_{\textrm{ref}}, c_{\textrm{ref}}}$ with all points in channel $c$, regardless of round. We adjust these points to be in the anchor coordinate system. In a formula, this reads as

$$
A_c = \textrm{ICP} (\textrm{base} = X_{r_{\textrm{ref}}, c_{\textrm{ref}}}, \quad  \textrm{target} = \bigcup _r B_r ^{-1} (\mathcal{F}_r ^{-1} (X_{r, c}))).
$$

The inverse transforms are used above because we are going from round $r$ coordinates to round $r_{\textrm{ref}}$ coordinates, which is opposite to the way we computed the transforms.

The chain of transforms is captured in the figure below:
<p align="center">
  <img src="https://github.com/user-attachments/assets/9362be6e-4b67-419b-a76e-661f98d84fef" width="450" />
  <br />
  <span> Chain of transformations learnt for each round and channel.</span>
</p>
