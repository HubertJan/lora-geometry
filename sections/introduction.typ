#import "@preview/pergamon:0.7.1": citep

= Introduction
Prior work in weight-space learning implicitly assumes that this space
is semantically organised. Adapter merging methods use the cosine
similarity between flattened task updates as a proxy for semantic
relatedness #citep("yadav2023ties") #citep("yu2024dare") #citep("stoica2025knots") #citep("zhao2024loralego"),
task arithmetic treats
weight deltas as vectors in an embedding space that can be added and
negated to compose capabilities #citep("ilharco2023task"), and meta models are
trained to read capability properties directly off adapter weights
#citep("unterthiner2020predicting") #citep("schurholt2021hyperrep") #citep("putterman2024lol")
#citep("sun2024peftguard") #citep("loracles2026"). If this assumption holds, a capability
should leave a weight-space signature that survives changes in how the
adapter was trained. To our knowledge this assumption has not been
directly tested.

We raise the question: _does a task capability leave a common
weight-space signature that survives changes in how the adapter was
trained, or is the finetuning weight-space organised by training
provenance instead?_

The practical stake is performance prediction: estimating how well an
adapter will score on a benchmark from its weights alone, without
running the benchmark. To answer the question, we train adapter pools
that vary systematically in training dataset and training conditions
--- 567 rank-16 adapters on SST2 #citep("socher2013sst2") and 2400 spanning 15 binary
classification datasets, all on Llama-3.2-1B --- and fit GL-equivariant
meta models #citep("putterman2024lol") that predict SST2 performance from
adapter weights alone. We evaluate them in-task, against hand-crafted
geometric baselines, and under leave-one-task-out distribution shift.
We then probe what the meta model reads using layer-wise relevance
propagation #citep("bach2015lrp") #citep("achtibat2024attnlrp"), causal keep-only ablation, and an analysis of the learned
probe directions.

We find that the weight-space is shaped mainly by the training setup
and datasets rather than by the capabilities it encodes. Within a
single-dataset pool, prediction is near-exact ($R^2 = 0.84$,
$rho = 0.95$ on held-out SST2 adapters) --- but a ridge regressor on a
handful of hand-crafted spectral and base-relative scalars already
recovers $R^2 = 0.64$, suggesting the signal sits in coarse properties
of the weight update rather than in capability-specific structure that
only a sophisticated probe could read. Under leave-one-task-out shift
only the ranking survives (median $rho = 0.66$) while calibration fails
outright ($R^2 < 0$ on 6 of 15 datasets). The interpretability analysis
is consistent with this: the adapter's capability is redundantly
encoded --- attention-only and MLP-only subsets each recover roughly
98% of the full adapter's performance lift --- and the directions the
meta model probes align with _where_ the base model writes its decision
rather than with the sentiment content it classifies. We therefore
argue that weight-space methods applied to LoRA adapters may exploit
correlations between training setup and resulting capability, rather
than capability structure in the weights themselves.

Concretely, we contribute: (i) two LoRA adapter pools with randomised
hyperparameters and injected label noise, designed to decorrelate
performance from training configuration; (ii) an evaluation of
GL-equivariant meta models against PEFTGuard #citep("sun2024peftguard") and
geometric-feature baselines, in-task and under leave-one-task-out
shift; and (iii) a component-level interpretability analysis of what
such a predictor reads, at the rank and scale used in practice --- 112
LoRA matrix pairs per adapter --- rather than in a minimal model
organism. #ref(<sec:approach>) describes the meta model and baselines,
#ref(<sec:setup>) the adapter pools and training protocol,
#ref(<sec:results>) the prediction and interpretability results.