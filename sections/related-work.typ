#import "@preview/pergamon:0.7.1": citep, citet

= Related Work

*Weight-space learning.*
Most weight-space work reads whole models rather than fine-tuned deltas:
predicting model characteristics from weights #citep("unterthiner2020predicting")
#citep("eilertsen2020classifying"), self-supervised weight representations
#citep("schurholt2021hyperrep") #citep("schurholt2022modelzoos"), and equivariant
weight-space architectures #citep("navon2023dws") #citep("zhou2023nfn").
Neural functional transformers #citep("zhou2023nft") and graph metanetworks
#citep("lim2024gmn") #citep("kalogeropoulos2024scalegmn") do handle
transformer-based models, but they build equivariance to operations _between_
model components rather than to the gauge freedom _inside_ a LoRA matrix pair, which base models do not exhibit. Such weight-space symmetries are a general
property of neural networks #citep("hechtnielsen1990algebraic")
#citep("entezari2022permutation") #citep("ainsworth2023gitrebasin")
#citep("godfrey2022symmetries"); the LoRA gauge freedom is the low-rank instance
of the same phenomenon. Learning-on-LoRAs (LoL)
#citep("putterman2024lol") is the exception, and the architecture we build upon.
Outside the weight-space field, backdoor detection has produced its own LoRA
meta models: PEFTGuard #citep("sun2024peftguard") treats the adapter as an
undifferentiated parameter tensor and is not gauge-equivariant, whereas the
spectral detector of #citet("puertolas2026weightspace") is invariant by
construction, reading only the singular values of $Delta W = B A$. Unlike this
line of work, we evaluate robustness under distribution shift in the training
data rather than in-distribution accuracy alone.

*The semantic-organisation assumption.*
Merging methods #citep("wortsman2022modelsoups") #citep("matena2022fisher")
#citep("yadav2023ties") #citep("yu2024dare") #citep("huang2024lorahub") assume
that weight-space proximity tracks semantics. LoRA-LEGO
#citep("zhao2024loralego") decomposes each adapter into per-rank units and
clusters them by similarity as a proxy for semantic relatedness, without
validating that proxy. KnOTS #citep("stoica2025knots") makes the assumption
explicit at the level of whole adapters, arguing that the pairwise cosine
similarity between flattened task updates is unreliable as a measure of merge
difficulty. Task arithmetic #citep("ilharco2023task") shows that the weight
space supports vector arithmetic on "weight vectors", but gives limited evidence
that independently trained updates sharing a capability end up close in that
space. Our results support this scepticism from a different direction: we find
that weight-space proximity tracks the training setup rather than the
capability.

*Interpreting individual adapters.*
LoRA #citep("hu2022lora") is the dominant parameter-efficient fine-tuning method
#citep("houlsby2019adapters"), effective because task adaptation has low
intrinsic dimensionality #citep("aghajanyan2021intrinsic")
#citep("li2018intrinsicdim"). Yet explanations of LoRA remain coarse — a
low-rank subspace update containing "intruder dimensions" absent from classical
fine-tuning #citep("shuttleworth2025illusion"), and it can learn less and forget
less than a full fine-tune #citep("biderman2024loralearnsless") — and
component-level attribution
has so far only been demonstrated in deliberately minimal settings, while
scalable mechanistic interpretability has otherwise concentrated on whole-model
features #citep("bricken2023monosemanticity") #citep("cunningham2024sparse").
LoRAcles
#citep("loracles2026") trains a language model that answers natural-language
questions about adapter weights, but its explanations describe the adapter's
training data rather than localising behaviour within the adapter.
#citet("soligo2025convergent") interpret a model organism built from nine rank-1
adapters and identify which of them carry narrow versus general misalignment
#citep("betley2025em") #citep("turner2025organisms"), and
#citet("rank1reasoning") treat a rank-1 adapter as a measurement device whose
per-component activations are individually interpretable. Both rely on each
adapter component reducing to a single scalar, which does not hold at the ranks
used in practice, and both analyse a handful of components by hand rather than
the full set. We instead work with rank-16 adapters spanning 112 matrix pairs,
ask which of those components a weight-space predictor actually relies on, and
evaluate that question under a distribution shift in the training data — a
setting none of the above address.