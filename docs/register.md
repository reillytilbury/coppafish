# Registration
The register section is the part of the pipeline concerned with aligning different rounds and channels. This is crucial for decoding spot colours into genes in the Call Spots and OMP sections. The aim of this section is to find a function $g_{trc}(\mathbf{x})$ for each tile, round and channel that takes in a location $\mathbf{x}$ on the anchor image for tile $t$ and returns the corresponding location in round $r$, channel $c$ of the same tile. Since registration is done independently for each tile and we are often only working on one tile, we sometimes omit the subscript $t$ in this documentation.

Once we have these transformations $g$, we can get a $n_{\textrm{rounds}} \times n_{\textrm{channels}}$ spot colours matrix $\boldsymbol{F}(\mathbf{x})$ for each location $\mathbf{x}$ in the anchor image of a given tile via

$$
\boldsymbol{F}(\mathbf{x}) =
\begin{pmatrix}
f_{0, 0}(g_{0, 0}(\mathbf{x})) & \cdots & f_{0, n_c}(g_{0, n_c}(\mathbf{x})) \\
\vdots & \ddots & \vdots \\
f_{n_r, 0}(g_{n_r, 0}(\mathbf{x})) & \cdots & f_{n_r, n_c}(g_{n_r, n_c}(\mathbf{x})) \\
\end{pmatrix},
$$

where $f_{rc}(\mathbf{x})$ is the intensity of round $r$, channel $c$ at location $\mathbf{x}$ in round $r$, channel $c$ coordinates. Note that even a single poorly aligned round or channel makes this matrix difficult to decode, which highlights the importance of this section.

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
D_{r_{\textrm{ref}}}(\mathcal{F}_r(\mathbf{x})) = D_r(\mathbf{x}).
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

which is now overdetermined! This is a better problem to have though, as the solution can be approximated by least squares.

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

    - as $\sigma \to \infty$ the image takes the same value everywhere, the mean of the flow image at the sample points.

    Another way of saying this is that as $\sigma$ grows, so does the radius of contributing pixels.

    We expect the shifts to vary more quickly in $z$ than in $xy$, so we have a different parameter for the blurring in each direction: `smooth_sigma`. This takes default values `[10, 10, 5]` ($y$, $x$ and $z$).
 

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

??? note "Definition of the Score $\lambda$"

    Define the auxilliary score 

    $$ 
    \eta(\mathbf{x}) = D_{r_{\textrm{ref}}}(\mathcal{F}_r(\mathbf{x}))D_r(\mathbf{x}).
    $$
    
    Then our score $\lambda$ is defined as

    $$
    \lambda(\mathbf{x}) = C_{0,1}\bigg( \dfrac{\eta(\mathbf{x}) - \eta_0}{\eta_1 - \eta_0} \bigg),
    $$

    where $\eta_0$ and $\eta_1$ are the 25th and 99th percentiles of $\eta$ respectively and $C_{a, b}$ is the [clamp function](https://en.wikipedia.org/wiki/Clamping_(graphics)).

    This results in a score of 0 for common low intensity background regions, and 1 for high quality regions like cell nuclei.
   
    

#### Extrapolation in z
The quality of the z-shifts drops rapidly towards the top end of the z-stack, because the optical flow uses windows of fixed radius (the `window_radius` parameter, which has default value 8). When these windows go over the edge of the image, the shifts get biased towards 0. This problem is made worse when the initial shift found is large in $z$, as then the adjusted image is padded with many zeros.

We get around this problem by linearly predicting the z-shifts from the bottom and middle of the image and replacing all z-shifts with these linear estimates. This is illustratedc in the figure below.
      
<p align="center">
<img src="https://github.com/user-attachments/assets/54d294f1-241c-4ea0-9f66-f700544fbd38" width="600" />
<br />
<span>Interpolation of x-y shifts, and extrapolation of z-shifts. </span>
</p>

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
# X = m x 4 matrix (base positions padded)
# Y = n x 3 matrix (target positions)
# transform_initial = 4 x 3 initial transform
# epsilon = distance threshold

# Initialize
transform = transform_initial
X = X @ transform
neighb_prev = None

# begin loop
for _ in range(n_iter):
    # Find closest point in X to each point in Y
    neighb = [argmin_k || X[k] - Y[j] || for j in range(n)]

    # Remove these matches if they are above the neighb_dist_thresh
    neighb = [neighb [j] if || X[neighb[j]] - Y[j] || < epsilon, 
              else None for j in range(n)]

    # Terminate if no change in neighbours
    if neighb == neighb_prev:
        QUIT

    # Update transform
    transform_update = argmin_B sum_j || X[neighb[j]] @ B - Y[j] || ** 2
    X = X @ transform_update
    transform = transform_update @ transform
    
    # Update neighb_prev
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


## Diagnostics
Problems in registration can ruin several downstream analyses. These problems can be  diagnosed by looking at the Registration Viewer, as follows:

```pseudo
from coppafish import Notebook, RegistrationViewer
nb_file = "path/to/notebook"
nb = Notebook(nb_file)
rv = RegistrationViewer(nb)
```

This will open a viewer with the following home screen:

<p align="center">
  <img src="https://github.com/user-attachments/assets/18f4c16a-9005-4808-acaa-e71fbeac3670" width="600" />
    <br />
    <span> The Registration Viewer </span>
</p>

This shows the round registration on the top row and the channel registration on the bottom row. This is displayed as follows:

- Each image in the top row shows a small patch of $(r_{\textrm{ref}}, c_{\textrm{dapi}})$ in red, overlaid with $(r, c_{\textrm{dapi}})$ in green. 

- Each image in the bottom row shows a small patch of $(r_{\textrm{ref}}, c_{\textrm{ref}})$ in red, overlaid with $(r_{\textrm{mid}}, c)$ in green.

Errors in registration may occur because of poor optical flow or ICP. The home screen shows small snippets of the images which indicate the overall quality of the registration. If these all look good, then the registration is likely fine. If not, then the options in the left panel will help diagnose the reason for poor round or channel registration.

### Different Methods

There are 2 sliders at the bottom of the viewer. The z-slider allows you to move through the z-planes, while the method slider allows you to choose between different methods of displaying the images. The methods are as follows:

1. **No Registration**: This shows the images without any registration. This is useful to see how big the misalignments are.

2. **Optical Flow**: This shows the images after the optical flow has been applied, but not the ICP transforms $A_{trc}$. The channel transforms shown here use the optical flow plus the initial affine transform $\tilde{A}_c$ learnt from the fluorescent bead images.

3. **Optical Flow + ICP**: This shows the images after the optical flow has been applied, and the ICP transforms $A_{trc}$ have been applied. This is the final registration.

!!! example "Registration with Different Methods"

    The figures below show a good example of the different stages of round registration. The largest changes are made by optical flow, while ICP makes smaller corrections.

    === "No Registration"
        <p align="center">
          <img src="https://github.com/user-attachments/assets/6da39f9f-2743-4e19-a075-19268a8c8411" width="600" />
         <br />
         </p>

    === "Optical Flow"
        <p align="center">
          <img src="https://github.com/user-attachments/assets/f748d2dd-3ab0-4127-b3dc-3399a2c1a47a" width="600" />
         <br />
         </p>

    === "Optical Flow + ICP"
         <p align="center">
          <img src="https://github.com/user-attachments/assets/ca9bd825-440f-46fc-9420-7988ed6593de" width="600" />
         <br />
         </p>

    The figures below show the different stages of channel registration. Here, image 1 is unregistered, image 2 is registered with optical flow and the initial affine transform, and image 3 is registered with optical flow and ICP. Most of the work done here is by ICP as the initial affine transform is not very good.

    === "No Registration"
        <p align="center">
          <img src="https://github.com/user-attachments/assets/b7d7de6c-69df-4c28-beea-e9fd3bda2182" width="600" />
         <br />
         </p>

    === "Optical Flow + Initial Affine Transform"
        <p align="center">
          <img src="https://github.com/user-attachments/assets/a5698e0b-05f8-4cc7-ae39-5774f75b6d39" width="600" />
         <br />
         </p>

    === "Optical Flow + ICP"
         <p align="center">
          <img src="https://github.com/user-attachments/assets/18f32ed5-637f-454b-b3aa-5103f7f29c09" width="600" />
         <br />
         </p>


### Optical Flow Diagnostics

The Optical Flow Viewer can be selected on the left hand panel to view the optical flow fields for a particular round. This will open a screen like the one below:

<p align="center">
<img src="https://github.com/user-attachments/assets/a5061aac-611b-4009-a32a-0d7097d3acdd" width="600" />
<br />
<span> The Optical Flow Viewer.</span>
</p>

This shows 3 columns of images:

1. **No Flow**: This shows $(r_{\textrm{ref}}, c_{\textrm{dapi}})$ in red, overlaid with $(r, c_{\textrm{dapi}})$ in green before optical flow has been applied.

2. **Raw Flow**: This shows $(r_{\textrm{ref}}, c_{\textrm{dapi}})$ in red, overlaid with $(r, c_{\textrm{dapi}})$ in green after the raw flow has been applied. 

3. **Smoothed Flow**: This shows $(r_{\textrm{ref}}, c_{\textrm{dapi}})$ in red, overlaid with $(r, c_{\textrm{dapi}})$ in green after the smoothed flow has been applied. 

Rows 2, 3 and 4 show the raw and smooth flows in the $y$, $x$ and $z$ directions respectively, while row 5 shows the correlation between the raw flow and the target image (this is the score $\lambda(\mathbf{x})$ which is used to compute the smoothed flow).

!!! example "Optical Flow Viewer Example"

    The figures below show an example of the different stages of optical flow. No flow shows a lot of misalignment. The raw flow shows the initial flow field, which is right in most places but wrong in others (particularly at edges). The smoothed flow is the final flow field, which is much better than the raw flow.

    === "No Flow"
        <p align="center">
          <img src="https://github.com/user-attachments/assets/0f74c38b-3480-4176-97f0-4ee23f735817" width="800" />
         <br />
         </p>

    === "Raw Flow"
        <p align="center">
          <img src="https://github.com/user-attachments/assets/8f7db328-14f4-45e2-a485-a210d43749f7" width="800" />
         <br />
         </p>

    === "Smoothed Flow"
         <p align="center">
          <img src="https://github.com/user-attachments/assets/24e2351e-c309-4f18-b19f-618a9be40d05" width="800" />
         <br />
         </p>
    
    The figure below is a closer look at the raw and smoothed flow fields, with the correlation plotted below them in blue.
    
    <p align="center">
    <img src="https://github.com/user-attachments/assets/c89c143b-646a-481f-aa13-188143620bf7" width="600" />
    <br />
    </p>


### ICP Diagnostics

Several diagnostics are available for ICP, and can be selected from the left hand panel. These viewers either show summary statistics or point clouds used to compute the transforms.

#### Summary Statistics
These show things like the average shift and scale for each round and channel and the convergence rates of each round and channel. This is useful for identifying outliers for some round or channel.

!!! example "Summary Statistics"

    The figure below shows the shifts and scales of the ICP correction for each round and channel of a particular tile. These numbers alone do not tell us the whole picture about the affine transforms (for example they don't tell us about the rotation), but they can be useful for identifying outliers, and seeing how much work ICP is doing.
    
    In this image, very bright or very dark columns indicate large round corrections, while very bright or very dark rows indicate large channel corrections. Take note of the following points:

    - The round corrections are largest in z. 
    - The channel corrections are largest in x and y.
    - The channel scales and shifts are very similar in channels separated by a multiple of 4. This is because these channels come from the same camera, and therefore have roughly the same offset.
    - Even though these scales are very small (around 1.003 at most), the images have size around 2000 pixels. This means that if we didn't correct for these scales, the images would be off by around 6 pixels, which is a lot.

    <p align="center">
    <img src="https://github.com/user-attachments/assets/e5447df4-77be-4d59-8bdf-43c21ffb4bf8" width="600" />
    <br />
    </p>
    

#### Point Clouds
These show the point clouds used to compute the round corrections $B_r$ and channel corrections $A_c$. This is much more detailed than the summary statistics and can be used to understand why convergence fails in certain cases.

!!! example "Point Clouds"

    The figure below shows the point clouds used to compute the channel correction $A_c$ for $c = 5$. 
    
    - The white circles are the points from $(r_{\textrm{ref}}, c_{\textrm{ref}})$,
    - the red crosses are the points from $(r_{\textrm{mid}}, c)$. 
    - The cyan lines show the matches between points in the unaligned point clouds,
    - the blue lines show the matches between points in the aligned point clouds. 
    - The yellow background image is bright in places where there are many matches and dark where there are few.
    
    === "No Registration"
        <p align="center">
        <img src="https://github.com/user-attachments/assets/151d275b-062b-4bcc-bbc6-26fcb1ab90ff" width="600" />
        <br />
        </p>

    === "Channel Correction"
        <p align="center">
        <img src="https://github.com/user-attachments/assets/aaf92759-92bc-416a-a5d5-a6ac14b614aa" width="600" />
        <br />
        </p>

    The figure below shows the point clouds used to compute the round correction $B_r$ for $r = 1$. This viewer has the same components as the channel correction viewer but it defaults to showing all z-planes, as this is what ICP corrects for the most.
    
    === "No Registration"
        <p align="center">
        <img src="https://github.com/user-attachments/assets/2df9bb7c-8967-4008-9cf9-12e957654752" width="600" />
        <br />
        </p>

    === "Round Correction"
        <p align="center">
        <img src="https://github.com/user-attachments/assets/5cbdbd4d-9373-4fb4-ac34-594203b1512b" width="600" />
        <br />
        </p>