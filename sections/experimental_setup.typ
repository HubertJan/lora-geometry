#import "@preview/pergamon:0.7.1": citep
#import "@preview/cetz:0.3.4"
#import "../loto-scatter.typ": famscatter
#import "../lrp-maps.typ": *

= Experimental Setup <sec:setup>
Both our method and the evaluation of any such method rely on a realistic, representative setup. We therefore take care to ensure the adapter pool is representative and not confounded.

#let basel2 = math.op("w128-l2")
#let base0 = math.op("w128")
#let bilin8 = math.op("w8-bilin-l2")
#let w8 = math.op("w8-l2")
#let w32 = math.op("w32-l2")
#let pg = math.op("PEFTGuard")
#let sridge = math.op("spectral-ridge")
#let nridge = math.op("norms-ridge")

== LLM LoRA Training
We exclusively fine-tune the Llama-3.2-1B base model (`meta-llama/Llama-3.2-1B`) #citep("grattafiori2024llama3") with LoRA at rank 16, using the PEFT #citep("mangrulkar2022peft") and Transformers #citep("wolf2020transformers") libraries. LoRA is applied to all 7 target modules (4 attention and 3 MLP components) across all 16 layers, giving $16 dot 7 = 112$ LoRA matrix pairs per adapter.

=== Target Tasks and Datasets
We train the LoRA adapters across the following binary tasks, summarised in @tab-tasks (appendix). The pool spans sentiment-polarity tasks (`sst2`, `imdb`, `rotten_tomatoes`, `yelp_polarity`, `amazon_polarity`), expected to transfer strongly to the `sst2` prediction target, and non-sentiment tasks: toxicity (`wiki_toxic`, `toxigen`, `civil_comments`, `wildguard`), entailment (`qnli`, `scitail`, `doc_nli`, `vitaminc`), and two individual datasets (`boolq`, `subj`) expected to transfer weakly and used to stress the provenance confound. All tasks are standardised to uniform "true"/"false" labels under a single descriptive chat template per task; the templates and the sampled LoRA training hyperparameters (@tab-hparams) are detailed in the appendix.

=== LLM Evaluation and Metrics
To evaluate training success and for the performance-prediction task itself, we score the trained models. All metrics are based on likelihood (rank) classification on the binary task using the held-out split of the respective dataset #citep("brown2020gpt3") #citep("gao2024lmeval"): we evaluate the likelihood of both response tokens ("true" and "false") and derive the metrics from these scores. From the resulting per-row class probabilities we compute six metrics, all on the SST2 test split and without additional forward passes, summarised in @tab-metrics.

#figure(
  caption: [The six SST2-test metrics derived from the per-row class probabilities.],
  table(
    columns: (auto, auto),
    align: (left, left),
    table.header(
      [*Metric*], [*Definition*],
    ),
    [Accuracy],        [Fraction of correctly classified examples.],
    [Macro-F1],        [Unweighted mean of per-class F1 scores.],
    [AUROC],           [Area under the ROC curve (ranking quality).],
    [Brier],           [Mean squared error between predicted probability and label #citep("brier1950verification").],
    [Mean-confidence], [Average predicted probability of the chosen class.],
    [NLL],             [Negative log-likelihood (classification test-loss).],
  ),
) <tab-metrics>

=== Adapter Pools
To investigate various properties of our model, we train large pools of adapters on the tasks described above using sampled hyperparameters. Each pool follows the same sampling procedure but differs in the datasets used to train it. An overview is given in @tab-pools; a per-dataset breakdown of the out-of-task (LOTO) pool is in @tab-pool-counts.

#figure(
  caption: [Two adapter pools are used across this report. The in-task SST2 pool has a dedicated
    train/test split; the out-of-task (LOTO) pool has no fixed split - under leave-one-task-out,
    the held-out dataset's adapters form the test set and the remaining adapters are used for
    training, rotating across datasets.],
  table(
    columns: (auto, auto, auto, auto),
    align: (left, left, center, center),
    table.header(
      [*Pool*], [*Datasets*], [*\#Adapters*], [*Train / Test*],
    ),
    [SST2 In-Task pool], [SST2],        [567],  [504 / 63],
    [LOTO pool],         [15 datasets], [2400], [-],
  ),
) <tab-pools>

// FLAG: origin of the 567 count (504 train + 63 test) is unexplained -- presumably the adapters
//       that passed QC / trained successfully. Add a one-line note once confirmed.
// TODO: briefly explain the in-task train/test split.

== Concrete Meta Model Architecture
Each adapter therefore presents $16 dot 7 = 112$ LoRA matrix pairs as input to the meta model, and all our meta models are built to consume adapters of this shape.

Throughout the experiments we instantiate this template at several concrete configurations, referred to by the abbreviations used in @tab-compare. The equivariant regressors differ mainly in the hidden dimension $d$ of the equivariant linear layer: $#w8$ ($d = 8$, $approx 811$k params), $#w32$ ($d = 32$, $approx 8.75$M params), and $#basel2$, the full dense head at $d = 128$ ($approx 123$M params). $#bilin8$ is the bilinear variant of the equivariant regressor at $d = 8$ ($approx 395$k params), which reads out directly from the bilinear form without a dense head. All four of these carry the $ell_2$-feature-normalisation described in @fig-lol-arch. To isolate the effect of that normalisation, $#base0$ is the same $d = 128$ dense head with the $ell_2$-feature-normalisation removed.

#figure(
  caption: [Training configuration shared by all meta models (regressors).],
  table(
    columns: (auto, auto),
    align: (left, left),
    table.header(
      [*Setting*], [*Value*],
    ),
    [Optimiser],          [Adam #citep("kingma2015adam")],
    [Learning rate],      [$2 times 10^(-5)$],
    [Weight decay],       [$1 times 10^(-5)$],
    [LR schedule],        [Cosine annealing #citep("loshchilov2017sgdr") ($T_"max" = 100$, $eta_"min" = 5 times 10^(-6)$)],
    [Epochs],             [$100$],
    [Batch size],         [$8$ (no gradient accumulation)],
    [Loss],               [Per-head MSE over the six metric heads, equal weight $1.0$, masked for missing targets],
  ),
) <tab-meta-hparams>

=== Meta Model Training
All meta models are trained with the same optimiser configuration, summarised in @tab-meta-hparams; only the training pool and split differ between the in-task and out-of-task experiments.
