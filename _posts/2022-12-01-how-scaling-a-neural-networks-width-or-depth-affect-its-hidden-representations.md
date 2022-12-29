---
layout: distill
title: How scaling a neural networks width or depth affect its hidden representations
description: todo....
date: 2022-12-01
htmlwidgets: true

# anonymize when submitting 
authors:
  - name: Anonymous 

# do not fill this in until your post is accepted and you're publishing your camera-ready post!
# authors:
#   - name: Albert Einstein
#     url: "https://en.wikipedia.org/wiki/Albert_Einstein"
#     affiliations:
#       name: IAS, Princeton
#   - name: Boris Podolsky
#     url: "https://en.wikipedia.org/wiki/Boris_Podolsky"
#     affiliations:
#       name: IAS, Princeton
#   - name: Nathan Rosen
#     url: "https://en.wikipedia.org/wiki/Nathan_Rosen"
#     affiliations:
#       name: IAS, Princeton 

# must be the exact same name as your blogpost
bibliography: 2022-12-01-how-scaling-a-neural-networks-width-or-depth-affect-its-hidden-representations.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
toc:
  - name: Introduction
  - name: Centered kernel alignment
  - name: Block structure
    subsections:
      - name: What happens within block structures?
      - name: Are block structures useful?
      - name: Collapsing the block structure
  - name: Cross model dynamics
  - name: Effect of depth and width on the models outputs
  - name: Discussion
---

## Introduction
When applying artificial neural networks, performance is usually optimized by varying the architecture depth and width.
There is a lack of understanding, however, what effect scaling the models main parameters - depth and width - has on
the learned representations. Do deep models learn different hidden layer features
than wide models? Also, are there systematical differences in outputs of deep and wide models?

This lack of insight is tackled in the paper

<p></p>
<span>&nbsp;&nbsp;&nbsp;&#9654;&nbsp;&nbsp;</span>Thao Nguyen et al. (ICLR, 2021) Do Wide and Deep Networks Learn the Same Things? Uncovering How Neural Network Representations Vary with Width and Depth <d-cite key="DBLP:conf/iclr/NguyenRK21"></d-cite>.
<p></p>

First, representation similarity is analysed between different layers of a single model. It is found that in overparameterized
models, so called <em>block structures</em> arise, which refer to groups of contiguous layers which have very similar hidden representations.
These block structures emerge, independent of whether a models width or depth is increased.\
Furthermore, it is shown that key components
of the representations are preserved and propagated during block structure layers. This opens various questions:

- Are representations becoming more meaningful for the task at hand, when being propagated through the block structure?
- Could block structure layers be pruned from the model without negatively affecting performance?
- What role do the residual connections, present in the models used in the experiments, play for keeping representations similar during propagation through the block structure?

Those questions are addressed by Nguyen et al. <d-cite key="DBLP:conf/iclr/NguyenRK21"></d-cite>, and their answers will be discussed and interactively visualized in the
following blogpost.

In addition, the blogpost presents novel findings described the second part of Nguyen et al. <d-cite key="DBLP:conf/iclr/NguyenRK21"></d-cite>'s analysis, which deal with representation similarities between different models, as well as whether models of different architecture type (e.g. deep vs. wide) show systematical differences in output predictions, despite performing very similar overall.

## Centered kernel alignment
Let's begin with having a look at methodology. It is not that easy to directly compare different layers representations,
for example because of their varying size and their distributed nature (meaning that important features can rely on different neurons
outputs). The tool Nguyen et al. <d-cite key="DBLP:conf/iclr/NguyenRK21"></d-cite> use for measuring representation similarity,
is called centered kernel alignment (CKA) <d-footnote>Note that Nguyen et al. <d-cite key="DBLP:conf/iclr/NguyenRK21"></d-cite> propose a slightly modified version of CKA in their paper, namely
minibatch CKA, which is used in their experiments. They modify classic CKA to reduce memory consumption, and their minibatch
CKA converges to the same value as CKA when the whole dataset is considered a minibatch.</d-footnote>, and addresses those problems.\
It follows a straight forward idea, which originates in neuroscience:
Similarity is not measured between representations directly, but between representation (dis-)similarity matrices
<d-cite key="kriegeskorte2008representational"></d-cite>.

<style>
#floated{
float: left;
}

</style>

When $\mathbf{X}$ is a matrix, holding the encoded features of size $m \times p_1$ ($m$ being the number of examples, $p_1$ being
the number of neurons), then $\mathbf{K}=\mathbf{XX}^\top$ refers to the $m \times m$ <ins>representation
similarity matrix</ins> of our first comparison layer. We can now compare how similar $\mathbf{K}$ is to a second representation
similarity matrix $\mathbf{L} = \mathbf{Y}\mathbf{Y}^\top$, with $\mathbf{Y}$ of size $m \times p_2$ holding the examples
encoded after our second comparison layer.\
In words, we do not compare representations, but how similar the relations between each layers representations are.


Proceeding to CKA, not a lot of logic is added. First, both representation similarity matrices have to be centered
(column and row means are being subtracted). This is done by computing $\mathbf{K}' = \mathbf{HKH}$, with $\mathbf{H}$ being
the centering matrix $\mathbf{H} = \mathbf{I}_n - \frac{1}{n} \boldsymbol{1}\boldsymbol{1}^\top$. In this case, $n$ equals $m$,
and $\mathbf{L}'$ is computed accordingly.
The similarity between $\mathbf{K}'$ and $\mathbf{L}'$ is then calculated using the Hilbert-Schmidt Independence Criterion (HSIC).
More specifically, $\textrm{HSIC} (\mathbf{K}, \mathbf{L})= \textrm{vec}(\mathbf{K}')\textrm{vec}(\mathbf{L}')/(m-1)^2$, which is
a measure invariant to orthogonal transformations, and thus to
permutations of neurons <d-cite key="DBLP:conf/icml/Kornblith0LH19"></d-cite>. Finally, CKA normalizes $\textrm{HSIC}$ to make the similarity
score invariant to isotropic scaling, and to get a nice range of values between 0 and 1.

Finally, the CKA formula is:

$$
\begin{equation}
\textrm{CKA}(\mathbf{K}, \mathbf{L}) = \frac{\textrm{HSIC} (\mathbf{K'}, \mathbf{L'})}{\sqrt{\textrm{HSIC} (\mathbf{K}, \mathbf{K})
\textrm{HSIC} (\mathbf{L}, \mathbf{L})}}.
\end{equation}
$$

## Block structure
The first part of the paper deals with representation structure within models when scaling their with and depth. Nguyen et al. <d-cite key="DBLP:conf/iclr/NguyenRK21"></d-cite>
train different variants of ResNets <d-cite key="DBLP:conf/cvpr/HeZRS16"></d-cite>, mostly on CIFAR-10, and calculate the CKA score for each pair of the models layers.
Those are then visualized in heatmaps as seen below. Three notes on this:
- The models compared within a graphic perform very similarly, with the maximum test accuracy margin between two models being 1,9%.
- The number of layers indicated in the plot might be bigger than the number of layers indicated in the models name.
This is due to the fact, that in the plots it is accounted not only for the convolutional layers,
but for all intermediate representations. Additional representations e.g. arise through pooling layers.
- The chessboard like structure arises as a result of the residual connections: Representations after a residual connection
are more similar to other post-residual representations, than to representations within a residual block.

The first major finding in the paper, is that with increased width or depth of models, blocks of contiguous layers with a very
high representation similarity arise. With the capacity of the models increasing, the blocks tend to get larger and more distinct.
One can see the phenomenon emerge in the graphic below:


<div id="images">
<img class="slide_pics" src="/public/images/2022-08-09-wide_vs_deep_network_representations/fig_1/deep_1.png" id="fig_1_deep" width="25%" height="25%">
<img class="slide_pics" src="/public/images/2022-08-09-wide_vs_deep_network_representations/fig_1/wide_1.png" id="fig_1_wide" width="25%" height="25%">

</div>

<div id="images">

  <div class="bar">
    <span class="slider_anno">Shallow</span>
    <input type="range" min="1" max="5" value="1" class="slider" id="slider_deep">
    <span class="slider_anno">Deep</span>
  </div>

  <div class="bar">
      <span class="slider_anno">Narrow</span>
      <input type="range" min="1" max="5" value="1" class="slider_small" id="slider_wide">
      <span class="slider_anno">Wide</span>
  </div>
</div>

<p align = "center" style="margin-top:20px">
<em>
Figure 1: The block structures emerge when increasing the models depth (left) or width (right).
</em>
</p>

<style>
.slide_pics {
    display: inline-block;
    margin-left: 3%;
    margin-right: 3%;
    margin-bottom: 5px;
    margin-top: 5px;
}

.slider_anno {
    font-size: 84%;
    font-family: Arial;
    color: black;
}

.bar {
    display: inline-block;
    margin-left: 32px;
    margin-right: 60px;
    margin-top: 0px;
}


#images{
    text-align:center;
}


.slider_small {
  -webkit-appearance: none;
  margin: 0 10px;
  width: 50%;
  height: 8px;
  border-radius: 5px;
  background: #d3d3d3;
  outline: none;
  opacity: 0.7;
  -webkit-transition: .2s;
  transition: opacity .2s;
}

.slider_small::-webkit-slider-thumb {
  -webkit-appearance: none;
  appearance: none;
  width: 15px;
  height: 15px;
  border-radius: 50%;
  background: #04AA6D;
  cursor: pointer;
}

.slider_small::-moz-range-thumb {
  width: 20px;
  height: 20px;
  border-radius: 50%;
  background: #5f079a;
  cursor: pointer;
}
</style>

<script>
//////////////////////////////////////////////////////////////////////// fig 1 deep
var slider_deep = document.getElementById("slider_deep");
var fig_1_deep = document.getElementById("fig_1_deep");

// output.innerHTML = slider_deep.value; // Display the default slider value

// Update the current slider value (each time you drag the slider handle)
slider_deep.oninput = function() {
if (slider_deep.value == 1) {
  fig_1_deep.src = "/public/images/2022-08-09-wide_vs_deep_network_representations/fig_1/deep_1.png"
}

if (slider_deep.value == 2) {
  fig_1_deep.src = "/public/images/2022-08-09-wide_vs_deep_network_representations/fig_1/deep_2.png"
}

if (slider_deep.value == 3) {
  fig_1_deep.src = "/public/images/2022-08-09-wide_vs_deep_network_representations/fig_1/deep_3.png"
}

if (slider_deep.value == 4) {
  fig_1_deep.src = "/public/images/2022-08-09-wide_vs_deep_network_representations/fig_1/deep_4.png"
}

if (slider_deep.value == 5) {
  fig_1_deep.src = "/public/images/2022-08-09-wide_vs_deep_network_representations/fig_1/deep_5.png"
}

}

//////////////////////////////////////////////////////////////////////// fig 1 wide
var slider_wide = document.getElementById("slider_wide");
var fig_1_wide = document.getElementById("fig_1_wide");

// output.innerHTML = slider_wide.value; // Display the default slider value

// Update the current slider value (each time you drag the slider handle)
slider_wide.oninput = function() {
if (slider_wide.value == 1) {
  fig_1_wide.src = "/public/images/2022-08-09-wide_vs_deep_network_representations/fig_1/wide_1.png"
}

if (slider_wide.value == 2) {
  fig_1_wide.src = "/public/images/2022-08-09-wide_vs_deep_network_representations/fig_1/wide_2.png"
}

if (slider_wide.value == 3) {
  fig_1_wide.src = "/public/images/2022-08-09-wide_vs_deep_network_representations/fig_1/wide_3.png"
}

if (slider_wide.value == 4) {
  fig_1_wide.src = "/public/images/2022-08-09-wide_vs_deep_network_representations/fig_1/wide_4.png"
}

if (slider_wide.value == 5) {
  fig_1_wide.src = "/public/images/2022-08-09-wide_vs_deep_network_representations/fig_1/wide_5.png"
}

}

</script>

A block of contiguous layers, with CKA scores close to one, is termed <em>block structure</em>. It is visible that block structures
arise independently of whether the models with or depth are increased.

Next, it is investigated whether the block structures emerge with regard to absolute model size, or
model size relative to the data. Nguyen et al. <d-cite key="DBLP:conf/iclr/NguyenRK21"></d-cite> proceed by training models, with gradually reduced training data,
while keeping the other model parameters static. The results can be explored below:

<div id="images">
    <img class="fig_2" src="/public/images/2022-08-09-wide_vs_deep_network_representations/fig_2/deep_1.png" id="fig_2_wide">
</div>

<div class="bar_centered">
    <span class="slider_anno">More data</span>
    <input type="range" min="1" max="3" value="1" class="slider" id="slider_fig2">
    <span class="slider_anno">Less data</span>

  <div id="button_fig2">
      <button onclick="changeDeepWide()" type="button" class="button" id="fig_2btn">Wide</button>
  </div>

</div>

<p align = "center" style="margin-top:20px">
<em>
Figure 2: Reducing the training data leads to block structures emerging in models with less capacity.
</em>
</p>

<style>
.button_fig2 {
    text-align:center;

}

.button {
    border: 2px solid black;
    font-size: 14px;
    background-color: #5f079a;
    border: none;
    color: white;
    padding: 12px 20px;
    text-align: center;
    text-decoration: none;
    display: inline-block;
    font-size: 16px;
    border-radius: 8px;
    cursor: pointer;
    width:80px;
    margin-bottom:20px;
}


.fig_2 {
    width : auto;
    height: 310px; /*to preserve the aspect ratio of the image*/
    display: inline-block;
}

.slider {
  -webkit-appearance: none;
  margin: 0 10px;
  width: 50%;
  height: 8px;
  border-radius: 5px;
  background: #d3d3d3;
  outline: none;
  opacity: 0.7;
  -webkit-transition: .2s;
  transition: opacity .2s;
}

.slider::-webkit-slider-thumb {
  -webkit-appearance: none;
  appearance: none;
  width: 15px;
  height: 15px;
  border-radius: 50%;
  background: #04AA6D;
  cursor: pointer;
}

.slider::-moz-range-thumb {
  width: 20px;
  height: 20px;
  border-radius: 50%;
  background: #5f079a;
  cursor: pointer;
}

.bar_centered {

  text-align:center;
  margin-top:0px;
}

.fig_2 {
  margin-bottom:0px;
}


</style>

<script>
//////////////////////////////////////////////////////////////////////// fig 2 wide
var wide = 2;
var slider_fig2 = document.getElementById("slider_fig2");
var fig_2_wide = document.getElementById("fig_2_wide");
var btn = document.getElementById("fig_2btn");


// output.innerHTML = slider_fig2.value; // Display the default slider value

// Update the current slider value (each time you drag the slider handle)
slider_fig2.oninput = function() {
if (slider_fig2.value == 1 && wide ==1) {
  fig_2_wide.src = "/public/images/2022-08-09-wide_vs_deep_network_representations/fig_2/wide_1.png"
}

if (slider_fig2.value == 2 && wide ==1) {
  fig_2_wide.src = "/public/images/2022-08-09-wide_vs_deep_network_representations/fig_2/wide_14.png"
}

if (slider_fig2.value == 3 && wide ==1) {
  fig_2_wide.src = "/public/images/2022-08-09-wide_vs_deep_network_representations/fig_2/wide_116.png"
}
if (slider_fig2.value == 1 && wide ==2) {
  fig_2_wide.src = "/public/images/2022-08-09-wide_vs_deep_network_representations/fig_2/deep_1.png"
}

if (slider_fig2.value == 2 && wide ==2) {
  fig_2_wide.src = "/public/images/2022-08-09-wide_vs_deep_network_representations/fig_2/deep_14.png"
}

if (slider_fig2.value == 3 && wide ==2) {
  fig_2_wide.src = "/public/images/2022-08-09-wide_vs_deep_network_representations/fig_2/deep_116.png"
}

}


function changeDeepWide() {
  if (wide == 1 && slider_fig2.value == 1){
    wide = 2;
    btn.firstChild.data = "Wide";
    fig_2_wide.src = "/public/images/2022-08-09-wide_vs_deep_network_representations/fig_2/deep_1.png";
    }

  else if (wide == 1 && slider_fig2.value == 2){
    wide = 2;
    btn.firstChild.data = "Wide";
    fig_2_wide.src = "/public/images/2022-08-09-wide_vs_deep_network_representations/fig_2/deep_14.png";
    }


  else if (wide == 1 && slider_fig2.value == 3){
    wide = 2;
    btn.firstChild.data = "Wide";
    fig_2_wide.src = "/public/images/2022-08-09-wide_vs_deep_network_representations/fig_2/deep_116.png";
    }

  else if (wide == 2 && slider_fig2.value == 1){
    wide = 1;
    btn.firstChild.data = "Deep";
    fig_2_wide.src = "/public/images/2022-08-09-wide_vs_deep_network_representations/fig_2/wide_1.png";
    }

  else if (wide == 2 && slider_fig2.value == 2){
    wide = 1;
    btn.firstChild.data = "Deep";
    fig_2_wide.src = "/public/images/2022-08-09-wide_vs_deep_network_representations/fig_2/wide_14.png";
    }

  else if (wide == 2 && slider_fig2.value == 3){
    wide = 1;
    btn.firstChild.data = "Deep";
    fig_2_wide.src = "/public/images/2022-08-09-wide_vs_deep_network_representations/fig_2/wide_116.png";
    }

}

</script>

It can be seen, that when reducing the data step by step, the blocks structures emerge in smaller models already,
for both deep and wide models. Therefore, it is concluded, that the emergence of the block structure seems to be an artifact
of overparameterized models.

### What happens within block structures?

After gaining knowledge about the block structure, the consequent follow-up question is, what happens to the representations within
the block structure. Revisiting how the CKA score was computed, note the two-step procedure that was used: First, representation similarity
matrices were computed (for each layer in this case), which were then compared with other representation similarity matrices.
This means that representations can numerically change between layers, however, the CKA score between those layers remains high
in case the relative representation structure remains similar.

So, what computations are done by the neural networks
during block structure layers? For investigating,
Nguyen et al. <d-cite key="DBLP:conf/iclr/NguyenRK21"></d-cite> look at a rewritten version of the CKA score. When $\mathbf{X}$ and $\mathbf{Y}$, of sizes $n \times p_1$ and
$n \times p_2$, hold the centered layer representations, the CKA score can be computed using the following formular:

$$
\begin{equation}
\textrm{CKA}(\mathbf{XX}^\top, \mathbf{YY}^\top) = \frac{\sum_{i=1}^{p_1} \sum_{j=1}^{p_2} \lambda_{X}^i \lambda_{Y}^j \langle
\mathbf{u}_{X}^i\,,\mathbf{u}_{Y}^j\rangle^2}
{\sqrt{\sum_{i=1}^{p_1} (\lambda_{X}^i)^2} \sqrt{\sum_{j=1}^{p_2} (\lambda_{Y}^j)^2}}.
\end{equation}
$$

The equation arises by rewriting $\mathbf{X}$ and $\mathbf{Y}$ in terms of their in terms of their singular value decompositions 
<d-cite key="DBLP:conf/icml/Kornblith0LH19"></d-cite>.
The vectors $$\begin{equation}\mathbf{u}_{X}^i\end{equation} $$ and $$ \begin{equation} \mathbf{u}_{Y}^j \end{equation} $$ refer to the
$$ \begin{equation} i^{\textrm{th}} \end{equation} $$/$$ \begin{equation} j^{\textrm{th}}\end{equation} $$ left-singular vector of $$ \begin{equation}
\mathbf{XX}^\top \end{equation}$$/$$ \begin{equation} \mathbf{YY}^\top \end{equation} $$, which refer to the normalized principle components
of $\mathbf{X}$ and $\mathbf{Y}$ <d-cite key="DBLP:conf/iclr/NguyenRK21"></d-cite>. Finally, $\lambda_{X}^i$ and $\lambda_{Y}^j$ are the corresponding squared singular values, which
measure the fraction of variance explained by each principal component in the representations.

Nguyen et al. <d-cite key="DBLP:conf/iclr/NguyenRK21"></d-cite> find, that the fraction of variance, that is explained by the first principal component,
is very high in network layers within a block structure, but low for other layers.

If for two layers, all the variance in the representations would be explained by their first principal components, the CKA score
between those layers would collapse to the squared alignment $$ \begin{equation} \langle \mathbf{u}_{X}^i\,,\mathbf{u}_{Y}^j\rangle^2\end{equation}$$
between those first principal components. It is thus suggested, that within block structure layers, where the CKA score
is continuously close to 1, the first principal component is <em>preserved</em> and <em>propagated</em>.

This theoretical finding is supported by analysing previously shown models with regard to the first principal component:


<div id="images">
  <img class="fig_3" src="/public/images/2022-08-09-wide_vs_deep_network_representations/fig_3/blocks.png" id="fig_3">

   <button onclick="changeBlockNoBlock()" type="button" class="button_f3" id="fig_3btn">No blocks</button>
</div>

<p align = "center" style="margin-top:20px">
<em>
Figure 3: Four plots, for four different models: A deep model with block structure (left), a deep model without block structure
(left, after clicking the 'No blocks' button), a wide model with block structure (right) and a wide model without block
structure (right, after clicking the 'No blocks' button"). On the top right for each model, the CKA heatmap is plotted as usual.
On the top left, the cosine similarity for each layers first principal component can be seen. The bottom left shows the
variance explained in the representations by the first principal component of each layer.
The bottom right shows the CKA heatmap for layers with the first principal component being deleted from the representation matrices.
</em>
</p>

<style>
.images{
    text-align:center;
}
.fig_3 {
    width : auto;
    height: 530px; /*to preserve the aspect ratio of the image*/
    display: inline-block;
    margin-left: 15px;
    margin-bottom: 0px;
}



.button_f3 {
    border: 2px solid black;
    font-size: 14px;
    background-color: #5f079a;
    border: none;
    color: white;
    padding: 12px 20px;
    text-align: center;
    text-decoration: none;
    display: inline-block;
    font-size: 16px;
    border-radius: 8px;
    cursor: pointer;
    width:130px;
    margin-bottom:20px;
    margin-top:0px;
    }

</style>

<script>
var fig_3 = document.getElementById("fig_3");
var btn2 = document.getElementById("fig_3btn");
var blocks = 1;

function changeBlockNoBlock() {
  if (blocks == 1){
    blocks = 2;
    btn2.firstChild.data = "Blocks";
    fig_3.src = "/public/images/2022-08-09-wide_vs_deep_network_representations/fig_3/no_blocks.png";
    }
  else if (blocks == 2){
    blocks = 1;
    btn2.firstChild.data = "No blocks";
    fig_3.src = "/public/images/2022-08-09-wide_vs_deep_network_representations/fig_3/blocks.png";
    }
  }
</script>

In the bottom left, we can see the fraction of variance explained in the representations by the first principal
component, per layer. Looking at the models exhibiting a block structure, it is very visible that the fraction of variance
explained within the block structures is greatly larger than in non block structure layers. When swapping to models without
a block structure, this difference becomes even more clear.

The top-left plot shows the cosine similarity between the first principal components. We can see that for models exhibiting a block structure,
the heat map looks very similar to the CKA score plotted top-right, which showcases that CKA reflects
the alignment between the first principal components, if the fractions of variance explained by them approach 1.
Looking at non block structure models, the CKA heatmap is visibly different from the heatmap comparing first principal components.

Finally, the CKA heatmap for models with the first principal component removed from the representations is shown at the bottom right.
For the overparameterized models, this removes the block structure from the heatmap. For the non block structure models,
the heatmap remains mostly unchanged.

"Together these results demonstrate that the block structure arises from preserving and propagating the first principal
component across its constituent layers." <d-cite key="DBLP:conf/iclr/NguyenRK21"></d-cite>.

Furthermore, all effects seem to be similar for the left and the right side, disregarding whether the models overparameterizations
come from increased width or depth.

### Are block structures useful?

While representations stay relatively similar in their relations to each other, when being propagated through the block structure,
one could still ask whether transformations applied within block structures impact task performance.\
For investigating this question, Nguyen et al. <d-cite key="DBLP:conf/iclr/NguyenRK21"></d-cite> train linear probes after each layer of different models. A linear probe resembles
a linear classifier, that maps directly from layers hidden representations to input examples labels <d-cite key="DBLP:conf/iclr/AlainB17"></d-cite>.
Linear probe accuracies can be seen below, for two models with block structure, and two without. Note that two models of
the same architecture type can exhibit different CKA heatmaps, due to their different initializations.

<div id="images">
  <img class="fig_4" src="/public/images/2022-08-09-wide_vs_deep_network_representations/fig_4/probes.png" id="fig_4_probes">
</div>

<p align = "center" style="margin-top:20px">
<em>
Figure 4: Top row: CKA heatmaps for models with different widths. Bottom row: Linear probe accuracy for each layer. The dashed
green lines refer to the boundaries between ResNet stages.
</em>
</p>

<style>

  .fig_4 {
    width : auto;
    height: 350px; /*to preserve the aspect ratio of the image*/
    display: inline-block;
  }
</style>

For the models without block structures on the left, we can see the accuracy of the linear probes monotonically increasing,
and with it the informativity of the hidden representations regarding the task at hand. For models exhibiting block structures on the right,
the picture looks more complex: For layers within a block, and before a residual connection, the linear probe accuracy
drops noticeably. For within block layers, after a residual connection, the accuracy increases only marginally.\
Also, there seems to be a jump in linear probe accuracy at the post-residual connection before the block structure, and one in
the very early layers, for both models on the right.
It feels though, that in overparameterized, block structure models, more of the relevant logic is performed in individual
layers. Other layers, especially during block structures, seem to mostly propagate information from previous layers. Furthermore,
this propagation seems to heavily rely on the residual connections present in ResNets, as pre-residual layers within blocks
seem to even worsen the representations.

For researching the dynamics between block structure emergence and residual connections, Nguyen et al. <d-cite key="DBLP:conf/iclr/NguyenRK21"></d-cite> trained plain
convolutional neural networks, without residual connections with varying widths, and computed the corresponding CKA heatmaps:

<div id="images">
  <img class="fig_1b" src="/public/images/2022-08-09-wide_vs_deep_network_representations/fig_1b/cnn1.png" id="fig_1b">
</div>

<div class="bar_centered_1b">
    <span class="slider_anno">Shallow</span>
      <input type="range" min="1" max="5" value="1" id="slider_fig_1b" class="slider_1b">
    <span class="slider_anno">Wide</span>
</div>

<p align = "center" style="margin-top:20px">
<em>
Figure 5: Block structures emerge in a model without residual connections with increased width.
</em>
</p>

<style>
.fig_1b {
    width : auto;
    height: 280px; /*to preserve the aspect ratio of the image*/
    display: inline-block;
    margin-bottom: 0px;
}

.bar_centered_1b{
  text-align:center;
  margin-bottom: 15px;

}
.slider_1b {
  -webkit-appearance: none;
  margin: 0 10px;
  height: 8px;
  border-radius: 5px;
  background: #d3d3d3;
  outline: none;
  opacity: 0.7;
  -webkit-transition: .2s;
  transition: opacity .2s;


}

.slider_1b::-webkit-slider-thumb {
  -webkit-appearance: none;
  appearance: none;
  width: 15px;
  height: 15px;
  border-radius: 50%;
  background: #04AA6D;
  cursor: pointer;
}

.slider_1b::-moz-range-thumb {
  width: 20px;
  height: 20px;
  border-radius: 50%;
  background: #5f079a;
  cursor: pointer;
}

</style>

<script>

var slider_1b = document.getElementById("slider_fig_1b");
var fig_1b = document.getElementById("fig_1b");

// output.innerHTML = slider_1b.value; // Display the default slider value

// Update the current slider value (each time you drag the slider handle)
slider_1b.oninput = function() {
if (slider_1b.value == 1) {
  fig_1b.src = "/public/images/2022-08-09-wide_vs_deep_network_representations/fig_1b/cnn1.png";
}

if (slider_1b.value == 2) {
  fig_1b.src = "/public/images/2022-08-09-wide_vs_deep_network_representations/fig_1b/cnn2.png";
}

if (slider_1b.value == 3) {
  fig_1b.src = "/public/images/2022-08-09-wide_vs_deep_network_representations/fig_1b/cnn4.png";
}

if (slider_1b.value == 4) {
  fig_1b.src = "/public/images/2022-08-09-wide_vs_deep_network_representations/fig_1b/cnn8.png";
}

if (slider_1b.value == 5) {
  fig_1b.src = "/public/images/2022-08-09-wide_vs_deep_network_representations/fig_1b/cnn10.png";
}

}
</script>

As shown for different ResNets, one can see blocks arise with increased model capacity. Based on the above graphic,
Nguyen et al. <d-cite key="DBLP:conf/iclr/NguyenRK21"></d-cite> conclude that the emergence of block structures does not seem to be affected by the residual connections.\
When carefully observing the pictures, however, there are differences to the block structures seen in the ResNets:
- The blocks are not as sharp at the borders, and also the CKA score within
the blocks is not uniformly as high as in the block structures observed in overparameterized ResNets.
- When further increasing the model capacity, after block structures already
emerged, blocks do not seem to change much anymore. Again, this is different to what was observed before in the ResNets, where the
position and especially the size of the block structures kept changing.

Finally, it still seems that the residual connections do play a role for the emergence, and especially nature of block structures.
More on this can be read in our discussion.

### Collapsing the block structure

We've seen that the block structure arises in overparameterized models, and also that it preserves and propagates key
components of the representations. Also, we've seen that the amount of task relevant information in the representations
barely rises during the block structure.\
Another way of researching whether/how much block structure layers are contributing in solving the final task,
is simply pruning block structure layers from the model, and see whether the final performance is affected.
Nguyen et al. <d-cite key="DBLP:conf/iclr/NguyenRK21"></d-cite> proceed to do exactly this, and for comparison, also non block structure layers are pruned from the models.\
More precisely, ResNet blocks are deleted one-by-one, starting at
the end of each ResNet stage. How this impacts the performance of models can be seen below, for two
models that exhibit block structure, and two that do not:

<div id="images">
  <img class="fig_5" src="/public/images/2022-08-09-wide_vs_deep_network_representations/fig_5/delete_blocks.png" id="fig_5">
</div>

<p align = "center" style="margin-top:20px">
<em>
Figure 6: Top row: CKA heatmaps for two narrow models (trained from different initializations) and two wide models.
Bottom row: Test accuracy after deleting blocks from the end of each ResNet stage (indicated by the green dashed lines),
while keeping the residual connections intact. The grey dashed line refers to the original models performance.
</em>
</p>

<style>
  .fig_5 {
    width : auto;
    height: 420px; /*to preserve the aspect ratio of the image*/
    display: inline-block;
  }
</style>

One can see that pruning blocks from the middle ResNet stage, and within a block structure, leads to only a small loss of
overall performance. Furthermore, the size of the block structure seems to play a role: the performance loss in the second
model from the right (large block structure) is even smaller than in the most right model (smaller block structure).
When pruning residual blocks from other parts of the models, as well as in models without block structures,
the performance loss is mostly pretty drastic.

After two experiments researching the dynamics between block structure layers and final task performance,
it forms the intuition that block structure layers seem to contribute little to the models overall performance.
In more pronounced block structures, the block structure layers contribution to final performance seems to approach
zero.

## Cross model dynamics
Next, the similarity of representations between different models is investigated.
Nguyen et al. <d-cite key="DBLP:conf/iclr/NguyenRK21"></d-cite> compare how similar the representations of architecturally identical models, with different initializations, and
how similar the representations of models with different architectures are. Again, the models compared exhibit a very similar testing accuracy.

For models without block structures, the CKA heatmaps between layers of different models look very similar to the heatmaps
between layers of the same model. This holds for when architecturally identical models are compared, as well as when
architecturally different models are compared. In the latter case, the representations are still similar along the diagonal
of the heatmap, suggesting similar representations at the same relative model depth.

For models exhibiting block structures, there is some representation similarity for non block structure layers.
However, when comparing layers of different models from which at least one layer is part of a block structure, in the
within model heatmap, there is nearly no to no representation similarity at all. Representations during block structures
therefore seem unique to each model.

## Effect of depth and width on the models outputs
Finally, Nguyen et al. <d-cite key="DBLP:conf/iclr/NguyenRK21"></d-cite> have investigated how model architectures and model outputs relate. They found that there are systematically
different output predictions made by different model types, even though the average accuracy is very similar.
The outputs are systematically different on class level, as well as on individual example level.
Let's have a look at the graphics below, to see those findings in detail:

<div id="images">
  <img class="fig_6" src="/public/images/2022-08-09-wide_vs_deep_network_representations/fig_6/fig_6.png" id="fig_6">
</div>

<p align = "center" style="margin-top:20px">
<em>
Figure 7: Left side: Comparison of example level accuracy of two groups of 100 networks each, trained on CIFAR-10. <strong>b</strong>:
Two groups of ResNet-62 models compared for showcasing the variance in example level predictions that occurs by chance.
<strong>a</strong>: A group of ResNet-14 (2x) compared with a group of ResNet-62s.
One can see that the variance is much higher than in the left bottom plot. <strong>c</strong>: Comparison of testing accuracies for different
groups of models trained on ImageNet, this time on class level. Orange dots: baseline comparison between ResNet-83 groups. Blue dots:
Group of ResNet-83s and ResNet-50 (x2.8).
</em>
</p>

<style>
  .fig_6 {
    width : auto;
    height: 500px; /*to preserve the aspect ratio of the image*/
    display: inline-block;
  }
</style>

On the left bottom side of the plot (<strong>b</strong>), we can see the average accuracy for individual examples, of two groups of 100 deep ResNets,
on the CIFAR-10 test set. Note that both groups have statistically indistinguishable average accuracies on the whole dataset.
The graphic acts as a baseline, to show what amount of variance between two groups of models should be expected by chance.
For comparison, on the top left (<strong>a</strong>), average individual example accuracies are shown for a group of wide models and a group of deep models.
We can see that the individual example accuracies vary much more than in the top left plot, which indicates that wide and deep
models make systematically different mistakes.

Insights in systematic differences in model outputs on class level, are shown on the right side of the figure.
This time, the class level accuracies of two groups of architectually identical deep models are compared as a baseline (orange dots),
with ImageNet being the dataset.
The blue dots compare a group of deep models and a group of wide models. Again, one can see that there are systematic differences
in output predictions, this time on class level. Looking at individual classes, three out of the five classes where the group
of wide models perform better, resemble scenes: seashore, library and book store. Following this intuition, Nguyen et al. <d-cite key="DBLP:conf/iclr/NguyenRK21"></d-cite>
find that the wide models significantly perform better on ImageNet classes that descend from "structure" or "geological formation".
Note that while the difference in accuracy is statistically significant, it is quite small
($74.9\% \pm 0.05 \ \mathrm{vs.} \ 74.6\% \pm 0.06$). Deep models on the other hand, performed significantly better on classes descending from
"consumer goods", again with a relatively small margin though ($72.4\% \pm 0.07 \ \mathrm{vs.} \ 72.1\% \pm 0.06$).

In the reviewing process, Nguyen et al. <d-cite key="DBLP:conf/iclr/NguyenRK21"></d-cite> hypothesize that wide layers might be better at capturing small details which would
help when detecting scenes, while depth would help when global structure is important, which would help with consumer goods.


## Discussion
Nguyen et al. <d-cite key="DBLP:conf/iclr/NguyenRK21"></d-cite> have shown that a so called block structure arises in overparameterized models, no matter if they are deep or wide.
A block structure consists of
many contiguous neural network layers, whose representations show very high similarity. They found that within a block structure,
the first principal component of the layers representation matrix is mainly propagated.\
They proceed by relating block structure and task performance, and find that the task-informativity of representations barely
rises during block structures. Also, it becomes visible that residual connections play an important role when propagating
information through block structure layers.\
Next, ResNet blocks were pruned from block structure models, and it was shown that block structure layers could be deleted
from models with minimal performance loss during the middle stage of the ResNet.\
When comparing different models representations, it was found that representations are similar regarding the relative depth of
models with different architecture types. Representations within block structures, however, didn't exhibit similarity with representations
from layers of other models.\
Finally, they look at the relation between model architectures and model outputs, and find that there are systematical
different mistakes made by models of different architecture types. This is observed on individual example level and on class level,
and even though the compared models performance is very similar on the whole dataset.

A first point of discussion, is the dynamic between residual connections and the block structure. While Figure 5 shows that a
block structure can also arise in a model without residual connections, there are differences in shape and delimitability.
More specifically, the block structure arising in the CNN seems less clear, and also, it doesn't increase size or clarity
when increasing the model capacity further and further. Also, the linear probe experiment shown in Figure 4
showcases the importance of residual connections for keeping the representations task relevance throughout the block structure.\
It remains a topic of research, thus, how well the findings about the nature of the block structure generalize to
different architecture types.
A hypothesis might be, that when unnecessarily increasing the model capacity of a ResNet, the flow of information
relies more and more on the residual connections, which propagate previously acquired logic in the representations.\
In contrast, the block structure layers of the CNN have to preserve the representations themselves, so e.g. a drop in task usefulness like seen for the pre-residual
layers within a block structure in Figure 4, would probably not be observed. A continuous (but probalby small) rise in linear
probe accuracy throughout the block structure CNNs seems more imaginable, and it would indicate a major difference in the
nature of block structure layers in ResNets and non-residual models.

Next, it remains the questions whether the block structure benefits
overall models performance. When looking at Figure 4 again, we can see that the linear probe accuracy still slightly increases in layers
within a block structure, after a residual connection. Furthermore, Nguyen et al. <d-cite key="DBLP:conf/iclr/NguyenRK21"></d-cite> present a table with each models test performance,
and while accuracy seems to saturate at some capacity, large models still perform better than smaller models in nearly all cases,
even though the large models exhibit massive block structures.\
Also, it was shown in Figure 6, that pruning layers from the block structure in the middle stage reduces accuracy, however,
only slightly. The accuracy seemed to decrease less, when the block structure was greater and more pronounced.\
It seems though, that some useful transformations are still going on within block structures, however, the transformations turn
less useful as the overparameterization increases.

For practical purposes of the presented findings, neural architecture search/model design would be the field to benefit:
Information about emerging block structures could be used to find a trade-off
between efficient, and successful neural architecture search. Detecting a block structures indicates being in the overparameterized regime,
meaning that training a less capacity model might yield a similar performance.

Also, a direction of thought could be designing networks that prevent themselves from building blocks, e.g. through
penalizing representation similarity in the loss. The idea would be, to still fight vanishing/exploding gradients using skip-connections
<d-cite key="DBLP:conf/cvpr/SzegedyLJSRAEVR15"></d-cite>, while preventing the net from simply propagating information during block structures.\
Whether this would yield a beneficial effect, or would solely take the self-regulating properties, brought by residual connections,
from the model, remains a topic for future research.

