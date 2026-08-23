#import "@preview/pergamon:0.7.1": citep
#import "/lol-arch.typ": lol-diagram

#let intrr = math.op("intrinsic-ridge")
#let baserr = math.op("baserel-ridge")
#let geomr = math.op("geom-ridge")

= Approach <sec:approach>

== Adapter Pools and Meta Models
In this report, we mainly work with LoRA fine-tuned LLMs, or LoRA adapters for short. When we refer to a LoRA adapter or LoRA weight, we typically mean just the fine-tuned weight deltas — the weights added to the already existing pre-trained model weights. When we discuss an adapter's capabilities, however, we always evaluate the full pre-trained model with those deltas applied.

We refer to a set of LoRA adapters, grouped by some common property, as a LoRA adapter pool - in other literature sometimes referred to as a model zoo #citep("schurholt2022modelzoos").

As part of the experiment, we train models that take LoRA adapters (i.e. weights) as input and produce some prediction as output. Such machine learning models are referred to as meta models. Different adapter pools serve as a proxy to evaluate the robustness of the meta model to distribution shifts: we evaluate whether the meta model is robust to distribution shift between the adapter pool used to train it and a hold-out adapter pool unseen at training time.

Evaluating robustness to various distribution shifts is relevant, as current weight-space methods mostly ignore detailed robustness evaluations.

== Workflow
The intended final workflow is as follows: we train various adapter pools under different settings and evaluate each on some task of interest. We then train a meta model on those adapter pools and their corresponding evaluation scores. The result is a meta model that can be used to evaluate new LoRA adapters.

== Meta Model <sec:meta-model>
The main focus of this report is a LoL-based weight-space classifier #citep("putterman2024lol"). It follows the GL-equivariant architecture approach by LoL #citep("putterman2024lol"). Given LoRA matrices $A, B$ it is equivariant to any transformations $M, N$ that transform $A, B$ as follows: $(B M)(N A)$. It has to hold for $M, N$ that $M N = I$. This equivariance is also rephrased as the equivariance to the gauge freedom $B -> B M, A -> M^(-1) A$ leaving the product $B A$ invariant. As LoRA adapter matrices can easily be transformed using this gauge freedom without breaking capabilities, this robustness to any such transformation is very relevant.


By default, the meta model processes the full adapter: all components at all layers. It can also operate on a subset of components, and it naturally handles adapters that were trained on only some layers or components. The equivariant (LoL) layers likewise extend to varying LoRA rank, so a single layer can process adapters of different ranks at once.



Each LoRA matrix pair is processed independently by the meta model. Each pair passes first through one equivariant linear layer #citep("putterman2024lol"), where all LoRA ranks of one LoRA matrix are processed by one shared matrix. Depending on the exact architecture applied, both matrices $B in RR^(m times r), A in RR^(r times n)$ are mapped down to $B' in RR^(d times r), A' in RR^(r times d)$ where $d < m$, $d < n$. Following the equivariant linear layer, the product $B'A'$ is formed of shape $d times d$. This product is then flattened to a vector $x_i$ of shape $d^2$. We apply an $ell_2$-feature-normalisation to each $x_i$; this normalisation proved necessary for training in many settings. All normalised $x_i$ across LoRA matrix pairs are concatenated into a single vector $x$. Finally, depending on the target metric, we apply a regression head with either a sigmoid activation, if the prediction target is bounded between 0 and 1, a tanh activation, if the prediction target is bounded between -1 and 1, or no activation, if the prediction target is unbounded.


#figure(
  lol-diagram,
  caption: [The LoL-based meta model architecture. Each LoRA matrix pair $B, A$ of a component (here
    Layer 1, Attention K) is reduced independently by a shared equivariant linear layer to
    $B', A'$. Their product $B'A' in RR^(d times d)$ is flattened and $ell_2$-normalised into a
    per-pair vector $x_i$. The vectors of all components are concatenated and passed to a
    single regression head that returns the prediction $hat(y)$.],
) <fig-lol-arch>

== Baselines
We consider several baselines. PEFTGuard #citep("sun2024peftguard") is originally a weight-space architecture for detecting backdoors in LoRA adapters; we adapt it to our regression tasks, as with the LoL-based models, by replacing its classification head with corresponding regression heads.

We further consider two very simple baselines. The first is a ridge regressor #citep("hoerl1970ridge") on the singular values of each LoRA matrix pair, where each pair contributes as many singular values as its LoRA rank. The second is a ridge regressor on the weight norms of all LoRA matrix pairs.

Beyond these, we examine more sophisticated geometric features. All features are computed per LoRA matrix pair on its effective update $Delta W = B A$, and we split them into two groups. The _intrinsic spectral features_ (@tab-metrics-intrinsic, appendix) are derived solely from the singular values (singular spectrum) $sigma$ of $Delta W$ and require no access to the base model. The _base-relative features_ (@tab-metrics-baserel, appendix) instead characterise $Delta W$ against the frozen base weight $W_0$. On top of each group we fit a ridge regressor ($#intrr$ and $#baserr$) and we fit $#geomr$ on the concatenation of both groups.
