# Call Spots Documentation

The Call Spots Page is a mode of gene calling which runs quickly on a small set of local maxima ($\approx$ 50, 000 per tile) of the anchor image. Initially, this was our final mode of gene calling, but has since been superseded by OMP. That being said, the call spots section is still a crucial part of the pipeline as it estimates several important parameters used in the OMP section.

Some of the most important exported parameters of this section are:
- **Bleed Matrix B:**  $(n_{\text{d}} \times n_{\text{c}})$ array of the typical channel spectrum of each dye,
- **Colour Scale Factor A:**  $(n_{\text{t}} \times n_{\text{r}} \times n_{\text{c}})$ array which multiplies the colours to minimise any systematic brightness variability between different tiles, rounds and channels,
- **Target Bled Codes K:**  $(n_{\text{g}} \times n_{\text{r}} \times n_{\text{c}})$ array of the expected colour spectrum for each gene

## Algorithm Flow Chart
![call_spots_colours](https://github.com/reillytilbury/coppafish/assets/88850716/21b77712-94cc-4c0b-ae20-e6b366c5826f)

## Algorithm Breakdown
The inputs to the algorithm are:
- Raw spot colours $F_{src}$ for all spots $s$ (defined as local maxima of round $r_{\text{anchor}}$, channel $c_{\text{anchor}}$)
- The tiles $t_s$ each spot $s$ was detected on
- The local coordinates $(x_s, y_s, z_s)$ for each spot
- A raw bleed matrix $\mathbf{B}$ of shape $(n_{\text{dyes}} \times n_{\text{c}})$ obtained from images of free-floating drops of each dye.

### 0: Preprocessing
We begin by transforming the raw spot colours via the following function: 
$$F_{s,r,c} \mapsto P_{t(s), r, c}F_{s,r,c} - \beta_{s,c}.$$ 
In the formula above:
- $F_{s,r,c}$ is the raw spot colour for spot $s$ in round $r$ and channel $c$, 
- $` P_{t,r,c} = 1/\text{Percentile}_s(F_{s, r, c}, 95) `$ for all spots in tile $t$ is the initial normalisation factor. Since $\approx 1/ 7$ of spots in round $r$ are expected to be brightest in channel $c$, this is normalising by the average intensity of a bright spot in this $(t, r, c)$, 
- $` \beta_{s,c} = \text{Percentile}_s(A_{t(s),r,c} F_{s, r, c}, 25)`$, where the percentile is taken across rounds. For 7 rounds, this is the second brightest entry of channel $c$ across rounds. This is a rough estimate of the background brightness of spot $s$ in channel $c$ after scaling by $\mathbf{A}$.

%TODO: Add an image of a spot before anything, after scaling, then after removing background

### 1: Initial Gene Assignment
The purpose of this section is to provide some gene assignments that will facilitate further calculations. To see why this is necessary, for many important variables we hope to calculate (eg: tile-independent bled codes for each gene $E_{g,r,c}$ or the tile-dependent bled codes $D_{g,t,r,c}$), we need samples of the genes to work with. This requires an initial assignment $g_s$ for each spot $s$. The simplest way to do this would be to:
1. Create an initial bled code $` \tilde{\mathbf{K}_g }= \mathbf{C_g B} `$ for each gene $g$ by matrix multiplying the code matrix $\mathbf{C_g}$ for each gene $g$ with the bleed matrix $\mathbf{B}$. (The code matrix $\mathbf{C_g}$  is a binary matrix of shape $(n_{\text{rounds}} \times n_{\text{dyes}})$ such that $C_{grd} = 1$ if gene $g$ has dye $d$ in round $r$ and 0 otherwise)
2.  Compute a similarity matrix $\mathbf{Y}$ of shape $(n_{\text{spots}} \times n_{\text{genes}})$ of the cosine angle of each spot $s$ to each gene $g$ defined by 
$$\mathbf{Y_{sg}}  = \mathbf{F_s \cdot \tilde{K_g}} = \sum_{rc}F_{src}\tilde{K_{grc}}.$$
3. Define a gene assignment vector $\mathbf{G}$ of length $n_{\text{spots}}$ defined by 
$$G_s = \text{argmax}  \_g Y_{sg}.$$

Unfortunately, this method does not work very well! The main reason for this is that the initial bled codes $\mathbf{\tilde{K}\_g}$ are very bad estimates of the true bled codes $\mathbf{K_g}$. Step 1 above assumes that each gene's expected code is just a copy of the expected dye in that round. In equations, this is saying

$$\mathbf{K_gr} = \mathbf{B_{d(g, r)}}.$$ 

However, due to random fluctuations in the concentrations of bridge probes for each gene in each round some genes appear systematically brighter/dimmer in some rounds! A much better model for the bled codes is $$\mathbf{K_gr} = \lambda_{gr} \mathbf{B_{d(g, r)}},$$

where $\lambda_{gr}$ is called the gene efficiency of gene $g$ in round $r$.  

Since we have no prior estimate of $\lambda_{gr}$, we need a method which will normalise each spot colour $F_{src}$ in each round, thereby removing any systematic scaling between rounds before proceeding in a way similar to steps 2 and 3. 

#### Probabilistic Gene Assignments using FVM
We are going to define a probability distribution of this spot $s$ belonging to dye $d$ in round $r$.
To that end, fix a spot $s$ and round $r$ and let $\mathbf{F_r}$ be this spot’s L2-normalized fluorescence vector (length $n_{\text{c}}$). Also let $\mathbf{B_d}$ be the L2-normalized fluorescence vector (length $n_{\text{c}}$) of dye $d$.

We'd like to assign a probability that any unit vector belongs to dye $d$. The simplest non-uniform distribution defined on the sphere is the [Fisher Von Mises distribution](https://en.wikipedia.org/wiki/Von_Mises%E2%80%93Fisher_distribution), which is just the restriction of a multivariate isotropic normal distribution to the unit sphere. This distribution has 2 parameters:
- A mean unit vector $\boldsymbol{\mu} \in \mathbb{S}^{n-1}$, 
- A concentration parameter $\kappa$ specifying how closely we expect the vectors to cluster around $\boldsymbol{\mu}$.

In our case $\boldsymbol{\mu}$ = $\mathbf{B_d}$ and $\kappa$ is a parameter that we leave arbitrary for the moment. Then the probability (density) of observing a fluorescence vector $\mathbf{f}$ given that that spot comes from dye $d$ is 

$$\mathbb{P}[\mathbf{F_r} = \mathbf{f_r} \mid D = d] = K \exp(\kappa\mathbf{f_r} \cdot \mathbf{B_d}),$$

where $K$ is a normalization constant we don’t need to worry about.

Write $\mathbf{F} = ( \mathbf{F_1}^T, \cdots, \mathbf{F_{n_r}}^T)$ for the $n_{\text{r}}n_{\text{c}}$-dimensional spot code with each round L2-normalized and $\mathbf{b_g} = (\mathbf{B_{d(g, 1)}}^T, \cdots, \mathbf{B_{d(g, n_r)}}^T)$ for gene $g$’s $n_{\text{r}}n_{\text{c}}$-dimensional bled code with each round L2-normalized. Assume that the rounds are statistically independent of each other. Then

$$ \mathbb{P}[\mathbf{F} = \mathbf{f} \mid G = g] = \prod_r K \exp(\kappa \mathbf{f_r} \cdot b_{g,r}) = \tilde{K} \exp \left( \kappa \sum_r \mathbf{f_r} \cdot \mathbf{B_{d(g, r)}}\right) =  \tilde{K} \exp(\kappa \mathbf{F \cdot b_g}) $$

By Bayes' Theorem:

$$ \mathbb{P}[G = g \mid \mathbf{F} = \mathbf{f}] = \dfrac{\mathbb{P}[\mathbf{F} = \mathbf{f} \mid G = g] \mathbb{P}[G = g]}{ \mathbb{P}[\mathbf{F} = \mathbf{f}]} $$

Assuming a uniform prior on the genes

$$ \mathbb{P}[G = g \mid \mathbf{F} = \mathbf{f}] = \frac{\exp(\kappa \mathbf{b}_g \cdot \mathbf{f})}{\sum_g \exp(\kappa\mathbf{b}_g \cdot \mathbf{f} )} =\text{softmax}(\mathbf{\kappa U f})_g, $$

where $\mathbf{U}$ is a matrix of shape $(n_g, n_r n_c)$ where each row $U_g = \mathbf{b_g}$. This shows that the value of $\kappa$ has no effect on the gene ordering, but just on the spread of probabilities among genes. A value of 0 yields a uniform distribution of probabilities between all genes, while a very large value of $\kappa$ is approximately 1 for the gene with the maximum dot product and 0 for all others.

### 3: Bayes Mean for Bled Code Estimation

The purpose of this section is to compute an average $(n_r, n_c)$ bled code for each gene $g$ that captures the relative brightness changes between rounds for each gene. We will make extensive use of the initial gene assignments, which are used to gather spots for each gene, and the bleed matrix we have computed, which forms the prior estimate of any fluorescence vector estimate. We will estimate tile-dependent bled codes, $\mathbf{D_{gtrc}}$ and tile-independent bled codes, $\mathbf{E_{grc}}$. By comparing these, we will be able to correct for tile by tile variations in brightness.

Our method of estimating $` \mathbf{E_{g,r}} `$ (and $` \mathbf{D_{g,t, r}} `$):

- Assumes a prior of $` \mathbf{E_{g,r}} = \mathbf{B_{d(g,r)}} `$, 
- Easily scales $`\mathbf{B_{d(g,r)}}`$ but needs many spots to change its direction,
- Allows some change in the direction if there are sufficiently many spots, meaning that we allow different genes to show the same dye differently. This is particularly relevant when dyes fail to completely wash out between rounds.

![bayes_mean_single](https://github.com/reillytilbury/coppafish/assets/88850716/1bfc7651-2967-4d00-94e9-4bfcc8a20ba2)
The Bayes Mean biases the sample mean towards a prior vector. This is useful when the number of samples is small and we expect the points to have mean parallel to the prior vector.

Fix a gene $`g`$ and round $`r`$ and let $` \mathbf{F_{1,r}}, \ldots, \mathbf{F_{n,r}} `$  be the round $r$ fluorescence vectors of spots assigned to gene $g$ with high probability.

We'd like to find a representative vector for this data which captures the mean length, but some genes have very few spots so an average is noisy. We therefore bias our mean towards a scaled version of $\mathbf{B}_{d(g,r)}$. 

To begin, assume the mean vector $` \overline{\mathbf{F}} = \frac{1}{n} \sum_{i} \mathbf{F}_{i,r} `$ is normally distributed and impose a normal prior on the space of possible means:

$$
\overline{\mathbf{F}} \sim \mathcal{N}(\boldsymbol{\mu}, I_{n_c})
$$

$$
\boldsymbol{\mu} \sim \mathcal{N}(\mathbf{B}_{d(g,r)}, \Sigma)
$$

where 

$$
\Sigma = \text{Diag}\left(\frac{1}{\alpha^2}, \frac{1}{\beta^2}, \ldots, \frac{1}{\beta^2}\right),
$$

in the orthonormal basis 
```math 
\mathbf{v}_1 = \mathbf{B}_{d(g,r)},
```
  
```math 
\mathbf{v}_2, \ldots, \mathbf{v}_n  \text{ orthogonal to } \mathbf{v}_1 .
```
We are going to let $\alpha << \beta$, which makes the set of probable means $\mathbf{\mu}$ an elongated ellipse along the axis spanned by $` \mathbf{B}_{d(g,r) } `$.

Define $\mathbf{Q} =\boldsymbol{\Sigma}^{-1}$ and $\mathbf{b} = \mathbf{B_{d(g,r)}}$. Since the normal is a conjugate prior, we know that the posterior $\boldsymbol{\mu} \mid \mathbf{\overline{F}}$ is normal. Thus finding its mean is equivalent to finding its mode. The log-density of $\boldsymbol{\mu} \mid \mathbf{\overline{F}}$ is given by

```math
\begin{aligned}
l(\boldsymbol{\mu}) &= \log P(\boldsymbol{\mu}| \overline{\mathbf{F}} = \mathbf{f}) \\ \\
       &= \log P(\boldsymbol{\mu}) + \log P(\overline{\mathbf{F}} = \mathbf{f} | \boldsymbol{\mu}) + C \\ \\
       &= -\frac{1}{2} (\boldsymbol{\mu} - \mathbf{b})^T Q (\boldsymbol{\mu} - \mathbf{b}) - \frac{1}{2} (\boldsymbol{\mu} - \mathbf{f})^T (\boldsymbol{\mu} - \mathbf{f}) + C
\end{aligned}
```

This has derivative

$$
\frac{\partial l}{\partial \boldsymbol{\mu}} = -Q (\boldsymbol{\mu} - \boldsymbol{b}) - (\boldsymbol{\mu} - \mathbf{f})
$$

Setting this to $\mathbf{0}$, rearranging for $\boldsymbol{\mu}$ and using the fact that

```math
Q \mathbf{v} = 
\begin{cases}
\alpha^2 \mathbf{v} & \text{if } \mathbf{v} = \lambda\mathbf{b} \\ \\
\beta^2 \mathbf{v} & \text{otherwise}
\end{cases}
```

we get

```math
\begin{aligned}
\mathbf{\hat{\mu}} &= (Q + I)^{-1}(Q \mathbf{b} + \mathbf{f}) \\ \\ 
    &= (Q + I)^{-1}(\alpha^2 \mathbf{b} + \mathbf{f}) \\ \\
    &= (Q + I)^{-1}(\alpha^2 \mathbf{b} + (\mathbf{f} \cdot \mathbf{b})\mathbf{b} + \mathbf{f} - (\mathbf{f} \cdot \mathbf{b})\mathbf{b}) \\ \\
    &= (Q + I)^{-1}((\alpha^2 + \mathbf{f} \cdot \mathbf{b})\mathbf{b} + \mathbf{f} - (\mathbf{f} \cdot \mathbf{b})\mathbf{b}) \\ \\
&= \dfrac{(\alpha^2 + \mathbf{f} \cdot \mathbf{b})}{1 + \alpha^2} \mathbf{b} +
 \dfrac{1}{1+\beta^2} \bigg( \mathbf{f} - (\mathbf{f} \cdot \mathbf{b})\mathbf{b} \bigg)
\end{aligned}
```
Plugging in $\mathbf{f} = \frac{1}{n}\sum_i \mathbf{F_{i, r}}$ yields our estimate $\mathbf{\hat{\mu}}$

![Bayes Mean](https://github.com/reillytilbury/coppafish/assets/88850716/01719532-4f2c-4710-acdd-6d6f74d7c892)
Decreasing $\beta$ increases the component of the Bayes Mean $\boldsymbol{\hat{\mu}}$ perpendicular to the prior vector. The values of $\alpha^2$ and $\beta^2$ should be thought of as the number of spots needed to change the scale and direction respectively of the prior vector.

### 4: Generation of Target Bled Codes $\mathbf{K_g}$

The purpose of this section is to try and remove systematic brightness differences between rounds and channels and to ensure that the dyes are well separated. 

To begin, fix a round $r$ and channel $c$ and let $d_{max}(c)$ be the dye which is most intense in channel c. We define a target value $T_d$ for each dye $d$ in its maximal channel $c_{max}(d)$. Now let $G_{r,d}$ be the set of genes with dye $d$ in round $r$, and define the loss function 

```math
L(V_{r, c}) = \sum_{g \in G_{r, \ d_{max}(c)}} \sqrt{N_{g}} \  \bigg( V_{r, c} \ E_{g, r, c} - T_{d_{max}(c)} \bigg)^2,
``` 
where $N_g$ is the number of high probability spots assigned to gene $g$. There is no reason this has to be a square root, though if it is not, too much influence is given to the most frequent genes. Minimise this loss to obtain

```math
V_{r, c} = \dfrac{ \sum_{g \in G_{r, \ d_{max}(c) }} \sqrt{N_g} E_{grc} T_{d_{max}(c)} } { \sum_{g \in G_{r, \ d_{max}(c) }} \sqrt{N_g} E_{grc}^2 },
```
which is our optimal value!

![scales](https://github.com/reillytilbury/coppafish/assets/88850716/83fca522-0f61-4a4f-b87f-16a466447146)
The gene intensities for each round and channel plotted in cyan, and their scaled versions plotted in red, showing how they have been recentred around the target values.

![scales_im (1)](https://github.com/reillytilbury/coppafish/assets/88850716/e6eb1bc9-59ff-4d2e-a39a-22eacee2d9d5)
The target scale matrix shows that most of its job is boosting channel 15 in this case, but the amount it boosts these values is highly variable between rounds.

Now define the _target bled codes_ 

```math
K_{g,r,c} = E_{g,r,c}V_{r,c},
```
which we will use instead of $E_{g,r,c}$ from here onwards.

### 5: Regression of $\mathbf{D_{g,t}}$ against $\mathbf{K_g}$

The purpose of this section is to remove brightness differences between tiles, and improve the round and channel normalisation we found in the previous step. We do this by finding a scale factor $A_{t, r, c}$ such that 
```math
A_{t,r,c} D_{g, t, r, c} \approx K_{g, r, c},
```
where $\mathbf{D_{g, t,}}$ is the tile-dependent bled code for gene $g$ in tile $t$ defined in step 3 and $\mathbf{K_g}$ is the target bled code for gene $g$ defined in step 4.

Our method works in a similar way to step 4: fix a tile $t$, round $r$ and channel $c$ and as above, let $G_{r,d}$ be the genes with dye $d$ in round $r$. Define the loss

```math
L(A_{t,r, c}) = \sum_{g \in G_{r, \ d_{max}(c)}} \sqrt{N_{g,t}} \  \bigg( A_{t,r, c} \ D_{g, t r, c} - K_{g, r, c} \bigg)^2,
``` 
where $N_{g, t}$ is the number of high probability spots of gene $g$ in tile $t$. Remember that $K_{g, r, c} = E_{g, r, c}V_{r, c},$ so writing this in full yields
```math
L(A_{t,r, c}) = \sum_{g \in G_{r, \ d_{max}(c)}} \sqrt{N_{g,t}} \  \bigg( A_{t,r, c} \ D_{g, t r, c} - E_{g, r, c}V_{r, c} \bigg)^2.
``` 
This means that if $D_{g,t,r,c} \approx E_{g,r,c}$ for all genes $g$ then $A_{t,r,c} \approx V_{r,c}$. This means that $\mathbf{A}$ is correcting for tile differences. Then why does it have indices for $r$ and $c$? Because the way that the brightness varies between tiles may be completely independent for different round-channel pairs. This is addressed further in the diagnostics. Minimising this loss yields:

```math
A_{t,r,c} = \dfrac{ \sum_{g \in G_{r, \ d_{max}(c) }} \sqrt{N_{g, t}} \ K_{g,r,c}  D_{g, t r, c}} { \sum_{g \in G_{r, \ d_{max}(c) }} \sqrt{N_{g, t}}  D_{g, t r, c}^2 }.
```
![homogeneous scale regression](https://github.com/reillytilbury/coppafish/assets/88850716/d3a01b61-21a5-4b88-bd43-821ca42d02c7)
We can use the `view_homogeneous_scale_regression` function to view the regression for a single tile. Note that the regressions seem to have a high $r^2$ value and the slopes are significantly different even within channels.

![different scales](https://github.com/reillytilbury/coppafish/assets/88850716/6e3e539c-0e9a-4a3d-a50e-fff19b29eeee)

### 6 and 7: Application of Scales, Computation of Final Scores and Bleed Matrix

The purpose of this step is to bring all the components together and compute the final scores and bleed matrix.

Now that we have the homogeneous scale $A_{t,r,c}$, we multiply it by the initial scale factor $P_{t, r, c}$ to get our final colour normalisation factor 

$$ A_{t, r, c} \mapsto A_{t, r, c} P_{t, r, c}.$$

This is important as all our calculations have been done on preprocessed spot colours which have already been multiplied by $\mathbf{P}$. We apply this scale to all of our spot colours $F$ by pointwise multiplication.

Next, we compute the final probabilities and dot product scores. We might ask at this point whether we should use:
1. The tile-independent bled codes $E_{g,r,c}$, 
2. the tile-dependent bled codes $D_{g,t,r,c}$ or,
3. the target bled codes $K_{g, r, c}$? 

The answer is definitely the target bled codes $K_{g, r, c}$! We calculated $\mathbf{E}$ and $\mathbf{D}$ as important summary statistics. These act as representative samples for each gene in the case of $\mathbf{E}$ and for each gene and tile in the case of $\mathbf{D}$. 

While it may seem more accurate to have a different gene code for each tile, the estimates in $\mathbf{D}$ are noisy due to small numbers of samples. Remember also that $\mathbf{A}$ was calculated specifically to maximise similarity of the **tile-dependent** codes $\mathbf{D}$ with the **tile-independent** target codes $\mathbf{K}$, meaning that multiplying spot colours by $\mathbf{A}$ has the dual effect of
1. homogenising them across tiles and,
2. bringing them all close to the target codes $\mathbf{K}$.

With that in mind, we compute:
1. Final gene probabilities using the scaled spots $\mathbf{AF}$ and comparing against the target codes $\mathbf{K}$
2. Final Dot Products using the scaled spots $\mathbf{AF}$ and comparing against the target codes $\mathbf{K}$. These would not have been accurate in step 1 as we had no model of how each gene varied in brightness between rounds, but now this is something we have accounted for in $\mathbf{K}$.
3. The Final Bleed Matrix using the same method as discussed in step 2, but with updated gene probabilities.