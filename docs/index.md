---
layout: default
title: Algorithmic String Art
---

<head>
  <link rel="stylesheet" href="assets/css/custom.css">
</head>

<script type="text/javascript"
  id="MathJax-script"
  async
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">
</script>

<!-- Glider.js CSS -->
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/glider-js@1/glider.min.css">

<!-- Glider.js JS -->
<script src="https://cdn.jsdelivr.net/npm/glider-js@1/glider.min.js"></script>

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/roy-hachnochi/string-art/blob/main/algorithmic_string_art_playground.ipynb)
[![GitHub](https://img.shields.io/badge/GitHub-View%20Repo-blue?logo=github)](https://github.com/roy-hachnochi/string-art)

---

# Algorithmic String Art

This is the project page for a personal weekend project, aimed at using computational optimization to recreate images by connecting strings through nails. The idea is inspired by the creatively brilliant work of Petros Vrellis: [A New Way to Knit (2016)](https://artof01.com/vrellis/works/knit.html). My vision was to make the algorithm as robust and automatic as possible, and although I can't say this was fully achieved, I did learn a lot along the way, and there are still plenty more interesting ideas to implement towards this goal.

In this post I will try to deep dive into the full details of the project development process and thought process, including failed ideas, efficiency considerations, and carefully deriving the mathematical formulas required to solve this problem.

<div style="text-align: center;">
  <video autoplay muted playsinline preload="auto" style="width=100%; max-width: 450px; height: auto; border-radius: 8px;">
    <source src="{{ site.baseurl }}/assets/images/inline/fish_MCBL_string_art.mp4" type="video/mp4">
  </video>
</div>

**[Setup and Simulation Model](#setup-and-simulation-model)**  
**[B&W Algorithms](#bw-algorithms)**  
**[Algorithm Improvements](#algorithm-improvements)**  
**[Multicolor Algorithms](#multicolor-algorithms)**  
**[Future Work](#future-work)**  
**[Examples](#examples)**

---

## Setup and Simulation Model
Our first course of action is to model the real-world String Art problem as a computer simulation. Doing this correctly is of vast importance, since the entire optimization process and rendering process are based on this.

### Single line simulation
Each line may be represented by a pair of nails on the canvas perimeter, our main challenge is how we simulate a line given this nail-pair. My first attempt was to explicitly model a line equation and thread profile, for example:  

$$
L(x,y) = 
\begin{cases}
1-\left(\frac{d(x,y;l)}{0.5t}\right)^2 & \text{:}\quad d(x,y;l) \leq t \\
0 & \text{:}\quad d(x,y;l) > t 
\end{cases}
$$  

Where $$d(x,y;l)$$ is the distance between pixel $$(x,y)$$ and the line $$l$$, and $$t$$ is the line thickness. This is very accurate and allows versatility in the line representation, but it's quite slow, as we must perform this calculation for every pixel in the image, and for every nail-pair.

Therefore, I turned to implement the lines with a library function. Specifically I found `skimage.draw.line_aa` to be the fastest implementation that supports antialiasing lines, which is important since the real-world strings aren't of constant intensity. This implementation also has the benefit of returning the image pixels which are included in the line, which will be important for efficient error calculation, since it will allow us to calculate the effect only on the relevant pixels.

<div class="image-captioned">
  <img src="{{ site.baseurl }}/assets/images/inline/line_const_vs_AA.jpg" alt="lines" style="width=100%; max-width:400px; display:block; margin:auto; border-radius:10px;">
  <div class="caption">Left: Constant line. Right: Antialiasing line, better resembling real world threads.</div>
</div>

### Canvas simulation
At first glance, it might seem as though using the highest canvas resolution is the best choice, but in practice using a smaller image size (around $$600\times600-900\times900$$) is better. This has two main advantages:  
1. **Line simulation time** - The less pixels there are to calculate, the less time it takes.
2. **Color blending** - Using a low resolution means that a string takes less than a pixel (accurately, a string's intensity through a pixel will be: $$I = width \cdot \frac{canvas\;resolution}{canvas\;size}$$). We haven't gotten to the colored version of the work, but having $$I<<1$$ will come in handy when simulating color blending of different color strings passing through the same pixel, which closely approximates how the human eye perceives colors.

However, for the final rendering we *will* use a bigger resolution, to simulate the real-world appearance of opaque overlapping strings, which don't blend when looking up close. Using the same formula as above, we'll set the canvas resolution so that the string intensity will turn out as $$1$$.

<div class="image-captioned">
  <img src="{{ site.baseurl }}/assets/images/inline/don_draper_full_vs_low_res.jpg" alt="canvas_res" style="width=100%; max-width:500px; display:block; margin:auto; border-radius:10px;">
  <div class="caption">Left: High resolution canvas. Right: Using opaque strings on a low resolution canvas doesn't simulate the real world details.</div>
</div>

---

## B&W Algorithms
Solving the black-and-white case is the core challenge. Multicolor optimization can either be reduced to, or built upon, a B&W base. I examined several approaches for this algorithm.

> #### A note on preprocessing
> Since the strings are black and the canvas is white, the first thing we do for each image is to negate it, making it's values $$1$$ for black and $$0$$ for white. Now the optimization process will work by adding blacks to recreate the image.

### Naive greedy optimizer
Although I called it "naive" due to it being the first method which comes to mind, it actually works surprisingly well! The greedy method works by minimizing the error for the current step only, and continuing from there. For our problem, this means that for each step we want to find the best line to add, i.e. the one which improves (minimizes) the error by the biggest factor. Putting this in mathematical terms, let $$x$$ be the current String-Art result, $$y$$ be the original image, and $$l$$ be an image representing the line being checked, we want:  

$$
\min_l \quad \nabla_l e = \left\| y-(x+l)\right\|_2^2 - \left\| y-x\right\|_2^2 \tag{1}
$$

> #### Two small notes
> 1. In practice, we calculate this only over the affected pixels, and not the entire image, for better efficiency.
> 2. Some might find it more intuitive to optimize the error directly: $$\min_l \quad e = \left\| y-(x+l)\right\|_2^2.$$ However, this is mathematically wrong, since what we really want is not to minimize the current error, but rather to perform gradient descent with respect to the added line, which is exactly what is represented in formula $$(1)$$.

<div class="image-captioned">
  <img src="{{ site.baseurl }}/assets/images/inline/don_draper_greedy_string_art.jpg" alt="greedy" style="width=100%; max-width:250px; display:block; margin:auto; border-radius:10px;">
  <div class="caption">Greedy optimizer</div>
</div>

### Linear optimizer
I love rigorously formulating problems as optimization problems. I find it to be one of the most elegant ways to apply the full power of mathematics, and once we succeed in doing so, a vast range of optimization algorithms are suddenly added to our toolbox. So naturally, this is what I wanted to do for our String-Art problem. The go-to approach would be a linear model.

$$
\min_x \quad \left\| Ax-b \right\|_2^2  \tag{2}
$$

Where we have:  
- $$b \in \left[ 0,1 \right]^{hw}$$ - The flattened image.
- $$x \in \left\{ 0,1 \right\}^{n}$$ - A binary vector representing which lines (from all possible nail-pairs) are added to the String-Art solution.
- $$ A \in \left[ 0,1 \right]^{hw \times n}$$ - The transformation matrix, which translates each line to it's image representation, and sums them to the result image.

We pre-calculate $$A$$ by using the canvas simulation for each possible nail pair, flattening the result, and stacking them as columns in the matrix.

> #### A note about efficiency
> The matrix $$A$$ is a very big one, and therefore consumes a lot of memory, and calculations involving it are time-consuming. However, we may notice that most of it's values are zeros (only the pixels related to the specific line in each column are nonzero), and choose to represent it as a **sparse** matrix, vastly improving both time and memory efficiency.

Now that we have this representation of the problem as a linear equation, we may plug in various linear regressors. The problem here, is that the solution isn't restricted to being binary (and not even restricted to being positive!). We may approximate the binary solution by setting some threshold, above which the values of $$x$$ will become $$1$$, and under which they will become $$0$$.

<div class="image-captioned">
  <img src="{{ site.baseurl }}/assets/images/inline/don_draper_LS.jpg" alt="LS" style="width=100%; max-width:500px; display:block; margin:auto; border-radius:10px;">
  <div class="caption">Left: Least Squares optimizer result. Right: After binarization.</div>
</div>

Unsurprisingly, although the optimizer solution was precise, the approximation is far from good, since the binarization process causes a divergence too big from the original non-binary solution.

So using Least-Squares optimization fails, but can this formulation help us develop something better, with just a little bit of hard work?

### Binary Linear optimizer

Let's try to combine the success of the greedy approach with the rigorous formulation of the linear model. What we want is a per-step update formula, derived from the minimization problem from equation $$(2)$$.

$$
\begin{align*}
e_{k+1} (l) &= \left\| A \left( x_k + I_l \right) - b \right\|_2^2  \tag{3} \\
&= \left\| \left( A x_k - b \right) + A I_l \right\|_2^2 \\
&= \left\| \left( A x_k - b \right) \right\|_2^2 + 2 \left( A x_k - b \right) \cdot A I_l + \left\| A I_l \right\|_2^2 \\
&= e_k + 2r_k \cdot A_l + \left\| A_l \right\|_2^2 \\
\end{align*}
$$

Where we defined:

$$
\begin{align*}
r_{k+1} (l) &= A x_{k+1} - b \\
&= A \left( x_k + I_{l_{k+1}} \right) - b \\
&= \left( A x_k - b \right) + A I_{l_{k+1}} \\
&= r_k + A_{l_{k+1}} \\
\end{align*}
$$

Such that $$l_{k+1}$$ is the next line number added to the solution, $$I_l \in \left\{ 0,1 \right\}^{n}$$ is a one-hot vector with $$1$$ in the index $$l$$, and $$A_l \in \left[ 0,1 \right]^{hw}$$ is the $$l$$-th column of $$A$$. And so, we arrive at a straight-forward update rule:

$$
l_{k+1} = \arg\min_l \left\{ e_{k+1} (l) = e_k + 2r_k^T A_l + \left\| A_l \right\|_2^2 \right\}  \tag{4}
$$

$$
r_{k+1} = r_k + A_{l_{k+1}} \tag{5}
$$

Put in words, in each step we calculate equation $$(4)$$ for all lines, find the best one, and update the residual error $$r_k$$ after adding this line via equation $$(5)$$. If we implement these equations using sparse matrices and vectorized calculations, this process becomes **extremely fast**!

<div class="image-captioned">
  <img src="{{ site.baseurl }}/assets/images/inline/don_draper_BL_string_art.jpg" alt="BL" style="width=100%; max-width:250px; display:block; margin:auto; border-radius:10px;">
  <div class="caption">Binary Linear optimizer</div>
</div>

The runtime improvement is also substantial. Compared on a $$300 \times 420$$ image:  

| Algorithm | Runtime |
| :---: | :---: |
| Greedy Optimizer | $$40$$ $$sec.$$ |
| Least Squares (Linear Optimizer)| ~$$11$$ $$min.$$ |
| Binary Linear Optimizer | $$5$$ $$sec.$$ |

---

## Algorithm Improvements
There are a few natural improvements to add to this algorithm, making it work better and faster.
1. **Continuous lines** - All of the above algorithms could output any sequence of nail-pairs, in any order. To make the physical weaving process easier, we'd like for the algorithm to work with a single long thread, so we should demand that each line starts from the target nail of the previous line. This is done by limiting equation $$(4)$$ to check only such valid lines. One way to do it is to set  $$e_{k+1}(l)=\infty$$ for non-valid lines.
2. **Importance weights** - In some images, where there is much detail in a specific region, we would like the algorithm to emphasize on these regions to capture those details. This may be done by adding a weight map $$W \in \left[ 0,1 \right]^{hw}$$ which gives higher weights to these regions. All it requires is a very simple manipulation (try developing the update formulas again yourself):
$$
A_W = W \odot A
$$,
where $$\odot$$ means element-wise multiplication of each column of $$A$$ with $$W$$.
3. **Valid nails subset** - We may make the algorithm even faster by limiting the number of nails we allow in each step. First, we don't want a nail to connect to nails which are too close to it anyways, since the result will be perceptually insignificant. Second, we limit it even further by selecting a smaller *random* subset of nails in each step. This hardly affects the output, but significantly improves efficiency.

<div class="image-captioned">
  <img src="{{ site.baseurl }}/assets/images/inline/don_draper_weights.jpg" alt="weights" style="width=100%; max-width:500px; display:block; margin:auto; border-radius:10px;">
  <div class="caption">Left: BL optimizer (no weights). Right: BL optimizer with weights, notice the enhanced details in the face and eyes.</div>
</div>

---

## Multicolor Algorithms
So we now have a very decent monochrome algorithm to recreate String-Art images, but why stop here? Although this part diverged from what my original intention for this project was, I wanted to make the most out of it.

Generalizing the monochrome algorithm to multicolor images is definitely not straight-forward. Given the subtractive nature of real-life colors, the naive method would be to apply the monochrome algorithm per channel, and render using CMYK (cyan, blue, magenta, black - the opposites of red, green, blue, white) threads. This works in theory and even in simulation, but when rendering with opaque strings it just completely fails.

<div class="image-captioned">
  <img src="{{ site.baseurl }}/assets/images/inline/fish_CMYK.jpg" alt="fish_CMYK" style="width=100%; max-width:500px; display:block; margin:auto; border-radius:10px;">
  <div class="caption">Optimizing CMYK channels separately. Left: Color subtraction simulation. Right: Real world opaque strings simulation.</div>
</div>

> #### A note about color theory
> I chose to model color blending in the straight-forward method - by simple addition in RGB space, but this is certainly not the most accurate color blending model there is. Our perception of colors (and color blending) is much more complicated, and modelling a correct color space is key for success of the multicolor algorithm. LAB for example, is considered a perceptual color space much more similar to how the human eye works.

In algorithms study, there are usually two main approaches to extend an algorithm to a broader/harder case:
1. **Reduction** - Taking the harder problem and breaking it into smaller, more manageable parts which we may already have the solution to.
2. **Generalization** - Looking at the base case as a specific case of a more general problem, and extending the idea of the base case to handle the general case.  

With this in mind, I'll break down how each approach leads us to different algorithms, both based on the monochrome algorithm which we already have.

### Reduction - Dithering + Monochrome Optimizer
After trying a few methods, and conducting some research, I came across this excellent [blogpost](https://www.perfectlynormal.co.uk/blog-computational-thread-art) by Callum Mcdougal, who made a very similar project (and from whom I borrowed most of my example images displayed here). He proposed a very interesting idea, which I had been circling around myself in my trials.

The key here is an image processing algorithm called Dithering, specifically [Floy-Steinberg dithering](https://en.wikipedia.org/wiki/Floyd%E2%80%93Steinberg_dithering). It is the process of approximating an image with a small color-palette, based on the idea that the human eye blends nearby colors. For example, a chess-board image with alternating black and white pixels will, from far enough away, just look gray. The algorithm works by scanning the image pixel by pixel, approximating the nearest color from the palette for each pixel, and diffusing the estimation error to the surrounding pixels.

<div class="image-captioned">
  <img src="{{ site.baseurl }}/assets/images/inline/dithering.jpg" alt="dithering" style="width=100%; max-width:500px; display:block; margin:auto; border-radius:10px;">
  <div class="caption">Left: Original. Right: Dithered.</div>
</div>

Building on this, we draw up the following method:
1. **Initialization** - Choose an appropriate color palette for the image (more on this later).
2. **Preprocessing** - Apply dithering to produce a single binary image for each color in the palette.
3. **Optimization** - Run the monochrome algorithm on each color-image separately.
4. **Postprocessing** - Interweave the threads of each color-image to a single solution.

For the postprocessing part, I came up with two methods:
1. **Interweave** - Based on Callum Mcdougal's idea, we:  
  a. Order the colors from lightest to darkest.  
  b. Reverse the order of threads for each color. This is because we assume that in the optimization process, the more important lines were discovered earlier, so we would like them on top.  
  c. Cyclically add only a segment of each color's threads (e.g., 25% of color #1, 25% of color #2, 25% of color #3, then the next 25% of each color until done).  
2. **Combine by simulation** - A different, optimization based, approach. Now that we have a fixed list of threads per color, we may solve a simpler optimization problem, which only chooses the next color to add in each step, progressing through the strings of each color in order. Since the search space is very small (only a single line to test per color) it runs pretty fast. Then just flip the order, placing the more important threads on top.

Personally, I couldn't decide which one worked better, but we may point out the difference between the approaches: **Interweave** is more user *controllable*, and **Combine** is more *automatic* but can sometimes turn out worst.

<div class="image-captioned">
  <img src="{{ site.baseurl }}/assets/images/inline/multicolor_BL_pipeline.jpg" alt="multicolor_pipeline" style="display:block; margin:auto; border-radius:10px;">
  <div class="caption">Multicolor optimizer pipeline</div>
</div>

### Generalization - Multicolor Binary Linear optimizer
The idea here is to rephrase our optimization problem (equation $$(3)$$) to the more general case of multicolor strings. Formally, we slightly change the definitions of the matrices and vectors, and add color representation:

- $$b \in \left[ 0,1 \right]^{hw \times 3}$$ - The flattened RGB image.
- $$C \in \left[ 0,1 \right]^{k \times 3}$$ - The color dictionary we're using, in RGB.
- $$x \in \left\{ 0,1 \right\}^{n \times k}$$ - A binary vector representing which lines (from all possible nail-pairs) are added to the String-Art solution, and in which color of the color dictionary. We allow at most a single $$1$$ per row.
- $$A \in \left[ 0,1 \right]^{hw \times n}$$ - The transformation matrix, doesn't change from it's original definition.

The new optimization problem becomes:

$$
e_{k+1} (l) = \left\| A \left( x_k + I_l \right) C - b \right\|_2^2 \tag{3'}
$$

And the update equations (this time I won't show the derivation):

$$
l_{k+1}, c_{k+1} = \arg\min_{l,c} \left\{ e_{k+1} (l) = e_k + 2C_c \cdot r_k^T A_l + \left\| A_l \right\|_2^2 \left\| C_c \right\|_2^2 \right\}  \tag{4'}
$$

$$
r_{k+1} = r_k + A_{l_{k+1}} \cdot C_{c_{k+1}} \tag{5'}
$$

With these update equations, we apply the exact same greedy iteration algorithm as the original monochrome solution, with each step adding a string of a new color.

> #### Implementing this in practice
> There are some software engineering technicalities here, such as how we enforce each color to start from it's previous nail, how we efficiently implement the calculations of equation $$(4'),$$ or how we make sure that we don't switch colors too often.

Finally, as in the previous approach, we assume that the more important lines are found earlier, so we flip the order in postprocessing.

<div class="image-captioned">
  <img src="{{ site.baseurl }}/assets/images/inline/fish_MCBL_string_art.jpg" alt="fish_MCBL" style="width=100%; max-width:300px; display:block; margin:auto; border-radius:10px;">
  <div class="caption">MCBL optimizer</div>
</div>

#### A slight improvement - CMY-log
Diving a bit deeper into color theory, I found that color blending behaves differently than just simple additive RGB blending. A more accurate model assumes **multiplicative** blending in CMY space (the complement of RGB). But how do we simulate multiplicative blending when all our equations above are linear? The good old $$log$$ trick will come in handy, as we find that we can just slightly change the inputs of the above optimization problem, and get the desired multiplicative behavior.

A single pixel color blending in this model is represented by:

$$
\begin{align*}
p_i &= 1 - \prod_{j} (1 - C_j)^{A_{ij}} \\
&= 1 - exp \left( \sum_{j} A_{ij} log \left( 1 - C_j \right)\right) \\
\end{align*}
$$

Therefore, we can change the inputs accordingly:

$$
\widetilde{C} = log(1 - C)
$$

$$
\widetilde{b} = log(1 - b)
$$

And from here we just solve the same optimization problem as before, with the updated inputs.

<div class="image-captioned">
  <img src="{{ site.baseurl }}/assets/images/inline/fish_MCBL_log.jpg" alt="fish_MCBL_log" style="width=100%; max-width:500px; display:block; margin:auto; border-radius:10px;">
  <div class="caption">Left: MCBL optimization on regular additive space. Right: MCBL optimizer on CMY multiplicative space.</div>
</div>

### Finding the Color Palette
We have just one last problem left to solve. Both methods heavily rely on choosing the correct color palette to approximate the image. From my experience, the algorithm is very sensitive to the chosen colors, and choosing even a slightly off palette will significantly degrade the results for most images. It's therefore very important to choose the best color palette in the preprocessing/initialization step.

Seeing as my aim was to have the algorithm be as much plug-and-play as possible, I tried three approaches, varying in levels of automaticity.
1. **Manual color palette** - The least automatic method, just receive a palette defined by the user. This has the best potential of being the most accurate, but finding the exact right palette may be hard and tiresome.
2. **Clustering-based palette** - Run a clustering algorithm (such as KMeans or Median-Cut) on the image pixels to choose the $$k$$ most dominant colors. This is the most automatic method, but unfortunately didn't work for most images since these algorithms tend to find average centers for clusters, and not the most common ones.
3. **RGBCMYKW-based palette** - Take a subset of $$k$$ colors between the 8 "corner" colors (red, green, blue, cyan, magenta, yellow, black, white). How do we choose the $$k$$ colors? We test all possible $$\binom{8}{k}$$ color combinations by dithering a few small windows of the image (thus making the search quick although it's brute-force). Note that with this method, we could also define a different color dictionary instead of RGBCMYKW.

I have a few ideas for other palette estimation methods, such as expanding option 2 to histogram-based approaches or expanding option 3 to use a finer color dictionary, but haven't gotten to trying them out yet.

<div class="image-captioned">
  <img src="{{ site.baseurl }}/assets/images/inline/fish_color_selection.jpg" alt="fish_color_selection" style="width=100%; max-width:800px; display:block; margin:auto; border-radius:10px;">
  <div class="caption">Left: Manual color selection. Middle: Color selection via clustering. Right: RGBCMYKW color selection.</div>
</div>

---

## Future Work
I'm winding down the work on this project for now, but here are a few directions I would love to explore in the future.
- **Better palette estimation method** - Having the algorithms success heavily rely on this part, I find it the most urgent step of the algorithm to improve. Maybe a histogram-based method would work here.
- **Better thread combination method** - The two proposed methods (interweave and combine) work pretty fine, but I do believe that better results can be achieved here. The *important = last* approach might also be suboptimal.
- **Deep Learning NN based algorithm** - Having a lot of background in the field of Deep Learning, as this is what I do in my day-job, I would really like to formalize the problem and solution as a neural network learning/optimization problem and give it a go sometime.
- **Better Robustness** - The aim of this project was for the algorithm to work automatically on _any_ image. This was only partially achieved as, from my experience, the algorithm works better on single or double color palette images, and finds it harder to achieve good results on full color images. This leads me to...
- **Get a deeper understanding on color blending** - The way we simulate the multicolor String-Art makes assumptions on color blending which might miss how this actually works in real life. In my simulation, I assume natural RGB-space blending, but in real life occlusions are to be considered, and color blending is probably more subtle (or at least modelled in a different color space). Using a better, more accurate, understanding of the physical model could open the door to working with more diverse, in-the-wild images.

I believe that this concludes the deep-dive into the process of developing my String-Art project. Although this has been a fairly long and thorough post, there are still some aspects of the project that I haven't covered, such as how we turn this into a PDF with instructions for physically making the String-Art, how we convert a non-continuous list of lines to a continuous one, how we initialize the canvas, and some other minor improvements I made in the algorithm.

Feel free to try the [Colab playground](https://colab.research.google.com/github/roy-hachnochi/string-art/blob/main/algorithmic_string_art_playground.ipynb), or reach out via [Email](roy.hachnochi@gmail.com) or [GitHub](https://github.com/roy-hachnochi/string-art) to bounce off ideas and improvements.


---

## Examples
<div class="glider-contain">
  <div class="glider">
    <div class="hover-effect"><div class="slide"><img src="{{ site.baseurl }}/assets/images/fish_MCBL_log_string_art.jpg" alt="fish"></div></div>
    <div class="hover-effect"><div class="slide"><img src="{{ site.baseurl }}/assets/images/tiger_BL_string_art.jpg" alt="tiger"></div></div>
    <div class="hover-effect"><div class="slide"><img src="{{ site.baseurl }}/assets/images/jellyfish_BL_string_art.jpg" alt="jellyfish"></div></div>
    <div class="hover-effect"><div class="slide"><img src="{{ site.baseurl }}/assets/images/stag_MCBL_log_string_art.jpg" alt="stag"></div></div>
    <div class="hover-effect"><div class="slide"><img src="{{ site.baseurl }}/assets/images/leopard_BL_string_art.jpg" alt="leopard"></div></div>
    <div class="hover-effect"><div class="slide"><img src="{{ site.baseurl }}/assets/images/lion_MCBL_log_string_art.jpg" alt="lion"></div></div>
    <div class="hover-effect"><div class="slide"><img src="{{ site.baseurl }}/assets/images/fish2_BL_string_art.jpg" alt="fish2"></div></div>
    <div class="hover-effect"><div class="slide"><img src="{{ site.baseurl }}/assets/images/fox_BL_string_art.jpg" alt="fox"></div></div>
    <div class="hover-effect"><div class="slide"><img src="{{ site.baseurl }}/assets/images/cat2_MCBL_log_string_art.jpg" alt="cat2"></div></div>
    <div class="hover-effect"><div class="slide"><img src="{{ site.baseurl }}/assets/images/cat_BL_string_art.jpg" alt="cat"></div></div>
    <div class="hover-effect"><div class="slide"><img src="{{ site.baseurl }}/assets/images/eye_BL_string_art.jpg" alt="eye"></div></div>
    <div class="hover-effect"><div class="slide"><img src="{{ site.baseurl }}/assets/images/earth_BL_string_art.jpg" alt="earth"></div></div>
    <div class="hover-effect"><div class="slide"><img src="{{ site.baseurl }}/assets/images/coraline_MCBL_string_art.jpg" alt="coraline"></div></div>
    <div class="hover-effect"><div class="slide"><img src="{{ site.baseurl }}/assets/images/duck_BL_string_art.jpg" alt="duck"></div></div>
    <div class="hover-effect"><div class="slide"><img src="{{ site.baseurl }}/assets/images/blade_runner_MCBL_string_art.jpg" alt="blade_runner"></div></div>
    <div class="hover-effect"><div class="slide"><img src="{{ site.baseurl }}/assets/images/london_telephone_box_MCBL_log_string_art.jpg" alt="london_telephone_box"></div></div>
    <div class="hover-effect"><div class="slide"><img src="{{ site.baseurl }}/assets/images/phoenix_BL_string_art.jpg" alt="phoenix"></div></div>
    <div class="hover-effect"><div class="slide"><img src="{{ site.baseurl }}/assets/images/planets_MCBL_log_string_art.jpg" alt="planets"></div></div>
    <div class="hover-effect"><div class="slide"><img src="{{ site.baseurl }}/assets/images/snake_MCBL_log_string_art.jpg" alt="snake"></div></div>
    <div class="hover-effect"><div class="slide"><img src="{{ site.baseurl }}/assets/images/volcano_BL_string_art.jpg" alt="volcano"></div></div>
    <div class="hover-effect"><div class="slide"><img src="{{ site.baseurl }}/assets/images/sauron_BL_string_art.jpg" alt="sauron"></div></div>
    <div class="hover-effect"><div class="slide"><img src="{{ site.baseurl }}/assets/images/mona_lisa_MCBL_log_string_art.jpg" alt="mona_lisa"></div></div>
    <div class="hover-effect"><div class="slide"><img src="{{ site.baseurl }}/assets/images/bee_MCBL_log_string_art.jpg" alt="bee"></div></div>
    <div class="hover-effect"><div class="slide"><img src="{{ site.baseurl }}/assets/images/pink_floyd_BL_string_art.jpg" alt="pink_floyd"></div></div>
    <div class="hover-effect"><div class="slide"><img src="{{ site.baseurl }}/assets/images/union_jack_MCBL_log_string_art.jpg" alt="union_jack"></div></div>
    <div class="hover-effect"><div class="slide"><img src="{{ site.baseurl }}/assets/images/H_BL_string_art.jpg" alt="H"></div></div>
    <div class="hover-effect"><div class="slide"><img src="{{ site.baseurl }}/assets/images/mona_lisa_BW_BL_string_art.jpg" alt="mona_lisa_BW"></div></div>
    <div class="hover-effect"><div class="slide"><img src="{{ site.baseurl }}/assets/images/don_draper_BL_w_string_art.jpg" alt="don_draper"></div></div>
    <div class="hover-effect"><div class="slide"><img src="{{ site.baseurl }}/assets/images/terminator_BL_string_art.jpg" alt="terminator"></div></div>
    <div class="hover-effect"><div class="slide"><img src="{{ site.baseurl }}/assets/images/joker_BL_string_art.jpg" alt="joker"></div></div>
    <div class="hover-effect"><div class="slide"><img src="{{ site.baseurl }}/assets/images/walter_white_BL_string_art.jpg" alt="walter_white"></div></div>
    <div class="hover-effect"><div class="slide"><img src="{{ site.baseurl }}/assets/images/kill_bill_BL_string_art.jpg" alt="kill_bill"></div></div>
    <div class="hover-effect"><div class="slide"><img src="{{ site.baseurl }}/assets/images/morrison_BL_string_art.jpg" alt="morrison"></div></div>
    <div class="hover-effect"><div class="slide"><img src="{{ site.baseurl }}/assets/images/godfather_BL_string_art.jpg" alt="godfather"></div></div>
    <div class="hover-effect"><div class="slide"><img src="{{ site.baseurl }}/assets/images/gatsby_BL_string_art.jpg" alt="gatsby"></div></div>
    <div class="hover-effect"><div class="slide"><img src="{{ site.baseurl }}/assets/images/einstein_BL.jpg" alt="einstein"></div></div>
    <div class="hover-effect"><div class="slide"><img src="{{ site.baseurl }}/assets/images/pulp_fiction_BL_string_art.jpg" alt="pulp_fiction"></div></div>
  </div>
  <button class="glider-prev">«</button>
  <button class="glider-next">»</button>
  <div role="tablist" class="dots"></div>
</div>

---

## References
- **[A New Way to Knit (2016)](https://artof01.com/vrellis/works/knit.html), Petros Vrellis** - The original String-Art project, which came up with this wonderful and creative idea, and inspired the work for everyone playing around with this problem.
- **[The Mathematics of StringArt](https://www.youtube.com/watch?v=WGccIFf6MF8&t=17s), Virtually Passed** - A fun and well-described youtube video, which was the first interaction I got with this line of work and inspired me to try it myself.
- **[Computational String Art](https://www.perfectlynormal.co.uk/blog-computational-thread-art), Callum Mcdougal** - An excellent blogpost I found while working on the project, which gave me some very clever and innovative ideas for the multicolor part, as well as most of the example images used in this page.
- **String Art: Towards Computational Fabrication of String Images, Birsak et. al.** - The best (and almost only) academic article I found related to the String-Art problem.



<script>
  window.addEventListener('load', function () {
    new Glider(document.querySelector('.glider'), {
      slidesToShow: 4,
      scrollLock: true,
      rewind: true,
      arrows: {
        prev: '.glider-prev',
        next: '.glider-next'
      }
    });
  });
</script>