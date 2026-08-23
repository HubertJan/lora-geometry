#import "@preview/pergamon:0.7.1": citep, citet
#import "@preview/cetz:0.3.4"
#import "../loto-scatter.typ": famscatter
#import "../lrp-maps.typ": *

#let basel2 = math.op("w128-l2")
#let base0 = math.op("w128")
#let bilin8 = math.op("w8-bilin-l2")
#let w8 = math.op("w8-l2")
#let w32 = math.op("w32-l2")
#let pg = math.op("PEFTGuard")
#let sridge = math.op("spectral-ridge")
#let nridge = math.op("norms-ridge")
#let intrr = math.op("intrinsic-ridge")
#let baserr = math.op("baserel-ridge")
#let geomr = math.op("geom-ridge")

// Per-cell (module, layer) LoRA site names for the UV analysis. Upright math ops.
#let attnOL11 = math.op("attnO-L11")
#let attnQL10 = math.op("attnQ-L10")
#let gateL13 = math.op("gate-L13")
#let downL12 = math.op("down-L12")
#let upL11 = math.op("up-L11")
#let attnVL1 = math.op("attnV-L1")
#let attnKL0 = math.op("attnK-L0")
#let gateL0 = math.op("gate-L0")
#let upL0 = math.op("up-L0")
#let downL1 = math.op("down-L1")
#let downL15 = math.op("down-L15")

// Per-cell decision-token read magnitude (arm B), sorted descending.
// fields: (label, tag, decision |vTx|, decision-is-strongest?, competing-category label)
#let _uvcells = (
  ("attnQ-L10", "",     0.232, true,  ""),
  ("up-L11",    "",     0.179, true,  ""),
  ("down-L1",   "ctrl", 0.162, true,  ""),
  ("attnO-L11", "",     0.125, true,  ""),
  ("attnV-L1",  "",     0.107, false, "BOS"),
  ("gate-L13",  "",     0.099, true,  ""),
  ("down-L12",  "",     0.079, false, "sent"),
  ("gate-L0",   "ctrl", 0.074, true,  ""),
  ("down-L15",  "last", 0.018, true,  ""),
  ("attnK-L0",  "ctrl", 0.005, false, "BOS"),
  ("up-L0",     "ctrl", 0.005, false, "punct"),
)
#let _uvnull = 0.037
#let _uvW = 9.0
#let _uvmax = 0.255
#let _uvxf(v) = v / _uvmax * _uvW
#let _uvchart = cetz.canvas({
  import cetz.draw: *
  let n = _uvcells.len()
  for t in (0.0, 0.05, 0.10, 0.15, 0.20) {
    let x = _uvxf(t)
    line((x, -0.5), (x, n - 0.4), stroke: (paint: luma(215), dash: "dotted", thickness: 0.4pt))
    content((x, -0.8), text(size: 7pt)[#t])
  }
  // random-direction null 95th percentile reference line
  line((_uvxf(_uvnull), -0.5), (_uvxf(_uvnull), n - 0.25),
    stroke: (paint: rgb("#c0392b"), dash: "dashed", thickness: 0.8pt))
  content((_uvxf(_uvnull), n - 0.15), anchor: "south",
    text(size: 6.5pt, fill: rgb("#c0392b"))[random-null p95 $= 0.037$])
  for (i, r) in _uvcells.enumerate() {
    let y = n - 1 - i
    let x1 = _uvxf(r.at(2))
    let col = if r.at(3) { rgb("#D55E00") } else { rgb("#7f8c8d") }
    rect((0, y - 0.32), (x1, y + 0.32), fill: col, stroke: 0.4pt + black)
    // value, and the winning category when the decision token is not strongest
    let vlab = if r.at(4) == "" { [#r.at(2)] } else { [#r.at(2) #text(fill: rgb("#7f8c8d"))[(#r.at(4))]] }
    content((x1 + 0.12, y), anchor: "west", text(size: 7.5pt, vlab))
    // cell name, with a small tag for control / last-layer cells
    let clab = if r.at(1) == "" { text(size: 8pt)[#r.at(0)] } else {
      text(size: 8pt)[#r.at(0) #text(size: 6pt, fill: luma(120))[#r.at(1)]]
    }
    content((-0.2, y), anchor: "east", clab)
  }
  content((_uvW / 2, -1.25), text(size: 8pt)[Decision-token $abs(v^top x)$ (norm-controlled)])
})



= Results <sec:results>
This section presents our results in two parts: first, performance prediction from adapter weights, and second, an interpretability analysis of what the predictor actually reads.

== Performance Prediction
We begin with performance prediction: estimating how well an adapter performs, once applied to the base model, without running the benchmark. Most methods we compare use only the adapter's own weights; some geometric-feature methods additionally use the base-model weights. Throughout, the prediction target is a performance metric on the SST2 task.

=== In-Task <in-task-eval>

#figure(
  placement: top,
  caption: [Held-out accuracy calibration ($R^2$) and ranking ($rho$, Spearman) on the shared
    63-adapter SST2 test set. Seed counts differ by model: $#w8$, $#bilin8$ and $#basel2$ are 4-seed means; $#w32$ and $#base0$ are single seed; $#pg$ is a 2-seed mean; the five ridge baselines ($#baserr$, $#geomr$, $#intrr$, $#sridge$, $#nridge$) are single fixed-split fits.
    // @ data/model_comparison.csv (n_seeds column); geom ridges from
    //   2026-08-24_geometry-baseline-perf-prediction (tierB/tierAB/tierA_ridge).
  ],
  table(
    columns: (auto, auto, auto),
    align: (left, center, center),
    table.header(
      [*Model*], [*Acc $R^2$*], [*Acc $rho$*],
    ),
    table.cell(colspan: 3, fill: luma(238))[*Equivariant regressors*],
    [$#w8$],    [0.844], [0.952],
    [$#w32$],   [0.840], [0.954],
    [$#bilin8$],[0.836], [0.946],
    [$#basel2$],[0.830], [0.954],
    [$#base0$], [$-0.080$], [0.632],
    table.cell(colspan: 3, fill: luma(238))[*Baselines*],
    [$#baserr$],[0.636], [0.869],
    [$#geomr$], [0.593], [0.855],
    [$#intrr$], [0.370], [0.861],
    [$#sridge$],[0.187], [0.820],
    [$#nridge$],[0.179], [0.814],
    [$#pg$],    [$-0.016$], [0.728],
  ),
) <tab-compare>

@tab-compare separates the models along two axes: ranking ($rho$) and calibration ($R^2$). Ranking is easy for almost every model: PEFTGuard ($0.728$) and the equivariant model without $ell_2$-normalisation ($0.632$) order the SST2 adapters reasonably well, and even the simple weight-statistics ridges reach $rho approx 0.82$. Calibration separates them sharply. The $ell_2$-normalised equivariant regressors predict accuracy nearly exactly ($R^2 approx 0.83$–$0.84$), and the $ell_2$-normalisation is doing the work: removing it collapses $R^2$ from $0.83$ to $-0.08$ while barely changing the ranking.

More telling for our argument is how well the hand-crafted geometric baselines calibrate. A ridge on base-relative geometric features alone reaches $R^2 = 0.636$, and on the combined feature set $0.593$ — far above the raw spectral and norm ridges ($0.187$, $0.179$) and above PEFTGuard ($R^2 approx 0$). That a handful of scalar weight-geometry features recovers most of the calibration signal suggests SST2 performance is legible directly from coarse properties of the weight update, not only from the equivariant model's learned representation. This is consistent with performance being tied to training-condition footprints in the weights rather than to a capability-specific structure that only a sophisticated probe could read.

=== Different Metrics

#figure(
  placement: top,
  caption: [Held-out $R^2$ (calibration) and $rho$ (Spearman ranking) for the two equivariant
    regressors $#w8$ (4-seed mean) and $#w32$ (single seed) across five of the six SST2 metrics (accuracy is in @tab-compare).
    // @ data/w8_w32_other_metrics.csv; source 2026-08-25_gl-regressor-complexity-ladder.
  ],
  table(
    columns: (auto, auto, auto, auto, auto),
    align: (left, center, center, center, center),
    table.header(
      [*Metric*],
      table.cell(colspan: 2)[*#w8* (811 k, 4 seeds)],
      table.cell(colspan: 2)[*#w32* (8.75 M, 1 seed)],
    ),
    [], [$R^2$], [$rho$], [$R^2$], [$rho$],
    [F1 (macro)],       [0.816], [0.952], [0.797], [0.955],
    [AUROC],            [0.878], [0.938], [0.864], [0.947],
    [Brier],            [0.912], [0.955], [0.906], [0.955],
    [Mean-confidence],  [0.956], [0.934], [0.951], [0.942],
    [NLL],              [0.910], [0.943], [0.912], [0.945],
  ),
) <tab-othermetrics>

As reported in @tab-othermetrics, no metric is a clear outlier: across both $R^2$ and $rho$, values fall roughly between $0.80$ and $0.96$. The one systematic pattern is along smoothness. The smoother, probability-based metrics  (Brier, mean-confidence and NLL) all reach $R^2 > 0.9$, whereas the harder, thresholded metrics (accuracy ($0.844$, @tab-compare), macro-F1 and AUROC) all sit below $0.9$. This is at least partly mechanical: the smoother targets are themselves more smoothly distributed across adapters, which makes them easier to calibrate against. Notably, mean-confidence is not a task-quality metric at all, yet it is predicted no better than the genuine performance metrics.

#let heat(v) = table.cell(
  fill: (if v >= 0 { rgb("#2c7fb8") } else { rgb("#d7301f") })
    .lighten(100% - calc.min(calc.abs(v), 1.0) * 78%),
)[$#v$]

#figure(
  caption: [Per-dataset out-of-task (LOTO) accuracy calibration $R^2$ and ranking $rho$ (Spearman)
    for $#basel2$, grouped by family. $R^2 <= 0$ marks well-ordered but mis-scaled predictions.
    Cells are shaded on a diverging scale centred at $0$ (blue positive, red negative;
    $abs("value")$ clamped at $1$). Single seed (seed 42); each row
    aggregates over the $160$ held-out adapters of that dataset.
    // @ data/base_l2_per_dataset_loo.csv all rows (acc_r2, acc_spearman); frozen from the sibling ood-nonsentiment-families base_l2 per-dataset LOTO R^2 + within-Spearman
  ],
  text(8pt, table(
    columns: (3.5cm, 1.55cm, 1.35cm, 1.05cm),
    align: (left, left, right, right),
    table.header([*Dataset*], [*Family*], [*acc $R^2$*], [*acc $rho$*]),
    [`imdb`], [sentiment], heat(0.597), heat(0.640),  // @ data/base_l2_per_dataset_loo.csv dataset=imdb
    [`rotten_tomatoes`], [sentiment], heat(0.594), heat(0.827),  // @ data/base_l2_per_dataset_loo.csv dataset=rotten_tomatoes
    [`yelp_polarity`], [sentiment], heat(0.589), heat(0.706),  // @ data/base_l2_per_dataset_loo.csv dataset=yelp_polarity
    [`amazon_polarity`], [sentiment], heat(0.010), heat(0.813),  // @ data/base_l2_per_dataset_loo.csv dataset=amazon_polarity
    [`wiki_toxic`], [toxicity], heat(0.179), heat(0.778),  // @ data/base_l2_per_dataset_loo.csv dataset=wiki_toxic
    [`toxigen`], [toxicity], heat(-0.079), heat(0.715),  // @ data/base_l2_per_dataset_loo.csv dataset=toxigen
    [`civil_comments`], [toxicity], heat(-0.172), heat(0.861),  // @ data/base_l2_per_dataset_loo.csv dataset=civil_comments
    [`wildguard`], [toxicity], heat(-0.204), heat(0.295),  // @ data/base_l2_per_dataset_loo.csv dataset=wildguard_prompt_harm
    [`qnli`], [entailment], heat(0.214), heat(0.499),  // @ data/base_l2_per_dataset_loo.csv dataset=qnli
    [`scitail`], [entailment], heat(-5.499), heat(0.280),  // @ data/base_l2_per_dataset_loo.csv dataset=scitail
    [`doc_nli`], [entailment], heat(-0.761), heat(0.301),  // @ data/base_l2_per_dataset_loo.csv dataset=doc_nli
    [`vitaminc`], [entailment], heat(0.165), heat(0.557),  // @ data/base_l2_per_dataset_loo.csv dataset=vitaminc
    [`boolq`], [qa], heat(0.186), heat(0.664),  // @ data/base_l2_per_dataset_loo.csv dataset=boolq
    [`subj`], [subjectivity], heat(-0.171), heat(0.408),  // @ data/base_l2_per_dataset_loo.csv dataset=subj
    [`sst2`], [sst2], heat(0.162), heat(0.805),  // @ data/base_l2_per_dataset_loo.csv dataset=sst2
  )),
) <tab-loo-perdataset>
=== Out-of-Task (LOTO)
To evaluate the robustness of the meta model on adapters trained on different data, we train meta models on various adapter pools and evaluate them on a hold-out adapter pool. Although the adapters are trained on differing tasks, they are all evaluated on the same SST2 task, and the meta model is trained to predict exactly this SST2 benchmark score. The per-dataset composition of this hold-out pool is listed in @tab-pool-counts.

Unlike the in-task results, where the meta model predicted the performance of unseen adapters well, this clearly fails in the unseen-dataset scenario. The per-dataset LOTO results are reported in @tab-loo-perdataset, with the sentiment family shown in @fig-loto-sentiment and the toxicity and entailment families in @fig-loto-toxicity and @fig-loto-entailment (appendix). The accuracy Spearman correlation stays relatively high (always above $0.25$, with mean $0.60$ and median $0.68$ across datasets) but the accuracy $R^2$ and the scatter plots show that the model does not calibrate to unseen datasets. The clear exceptions are three of the four sentiment datasets (`imdb`, `rotten_tomatoes` and `yelp_polarity`) which each reach accuracy $R^2 > 0.5$. `amazon_polarity`, though also a sentiment task, is the exception to the exception ($R^2 = -0.075$): it ranks well ($rho = 0.773$) but does not calibrate. The three that do generalise are plausibly similar enough that adapters from one transfer to the others; whatever sets `amazon_polarity` apart, that similarity does not extend to it.

#figure(famscatter("sentiment", size: (5.6, 4.6)),
  caption: [Sentiment family — $#basel2$ LOTO predicted vs. true SST2 accuracy, per dataset.
    Toxicity and entailment counterparts are in the appendix (@fig-loto-toxicity, @fig-loto-entailment).]
) <fig-loto-sentiment>

== Interpretability
As the out-of-task (LOTO) meta models showed rather weak performance, we concentrate in the following on the meta model trained for @in-task-eval.

=== LRP Maps <sec:lrp>
To reduce the noise between meta model seeds, we only look at average Layerwise-Relevance-Propagation (LRP) #citep("bach2015lrp") #citep("montavon2019lrpoverview") heatmaps in the following, averaged over the four $#w8$ meta-model seeds (42--45) across the 63 test adapters. We use the attention-aware propagation rules for transformers #citep("ali2022xaitransformers") #citep("achtibat2024attnlrp").
// FACT-CHECK: dropped the sentence "We analysed the seed noise ... in the appendix" -- no such
// seed-noise figure/analysis exists in the appendix (see the TODO by @tab-pool-counts). The source
// experiment 2026-08-25_gl-regressor-lrp-acc DOES carry a seed-stability analysis
// (plot_fig-seed-stability.py / plot_fig-seed-examples.py) that could be ported here; until it is,
// the forward reference would be unsupported.

We mainly focus on a component-level aggregation visualisation of those LRP scores. For that, we compute the LRP scores for each individual weight of the LoRA adapter. Then, we aggregate by addition all LRP relevances inside one LoRA adapter matrix. So negative and positive relevance might balance each other out. As, by the LRP conservation property #citep("montavon2019lrpoverview"), LoRA A and LoRA B receive the exact same relevance, this relevance is multiplied by 2 to obtain the LoRA matrix pair relevance, or, in the context of the whole adapter, the relevance for a certain component at a certain layer.

#figure(
  stack(spacing: 9pt,
    align(center, stack(spacing: 3pt, text(size: 8pt, weight: "bold")[(a) PCA scree — PC1 $= 92%$], _screecanvas)),
    align(center, stack(spacing: 3pt, text(size: 8pt, weight: "bold")[(b) pairwise map cosine (bimodal)], _histcanvas)),
  ),
  caption: [Diversity of the #w8 accuracy-head LRP maps across adapters. *(a)* PCA scree: PC1 alone
    explains $92%$ of the between-map variance. *(b)* the distribution of pairwise map cosines is
    bimodal.
    // @ data/pca_variance.csv (PC1 = 0.9159), data/pairwise_cos_hist.csv; source 2026-08-25_gl-regressor-lrp-acc.
  ],
) <fig-lrp-diversity>

As seen in @fig-lrp-diversity, the LRP maps are in general very similar across adapters. 92% of the variance between those LRP maps can be explained using the first PC. PC1 and the PCA mean are visualised in @fig-lrp-pca (appendix).

Noteworthy about the LRP relevance distribution is that the strongest high-vs-low accuracy contrast (the PC1 loading, @fig-lrp-pca) is concentrated in the early to middle layers, roughly layer 1 to layer 10, in just the MLP components. Meanwhile, the later layers are much more important for low accuracy adapters.
// FACT-CHECK: reworded. The raw high-accuracy relevance map is fairly uniform across layers
// (down_proj ~0.008-0.025, if anything slightly late-concentrated); what IS concentrated in the
// early-middle MLP layers is the high-vs-low PC1 contrast (down_proj positive ~L1-L10, already
// negative by L11-L12). @ data/pc1_loading.csv. Source: 2026-08-25_gl-regressor-lrp-acc. The attention components K and V are of particularly little importance, as, in particular in the layers up to layer 11, the relevance is around 0 across all adapters.

High-accuracy maps are far more mutually similar than low-accuracy ones, which are in turn more similar to each other than to the high-accuracy cluster (@fig-lrp-clusters, appendix).


// ============================================================================
// Native-Typst module-group sufficiency bar chart (drawn with cetz primitives).
// Horizontal bars = the SST2 accuracy that one KEPT module-group alone reconstructs,
// most-sufficient group at the top, per-adapter dots overlaid, dashed base-floor (0 % of
// lift) and full-adapter (100 %) reference lines. Data are the keep-only SST2 accuracies
// frozen in data/module_group_sufficiency.csv (3 high-accuracy adapters), selected from
// the sibling experiment 2026-08-25_kv-ablation-high-acc (its Fig. 2 sufficiency spectrum).
// ============================================================================
#let _mscolors = (reference: rgb("#4d4d4d"), attention: rgb("#3b6fb0"), mlp: rgb("#e08a3c"))
// FLAG: leading-slash path "/data/..." is root-relative in Typst; every other data reference in
//       this file is written relative ("data/..."). Verify this resolves given the file's location
//       (imports use "../"), i.e. whether it should be "../data/module_group_sufficiency.csv".
#let _msrows = csv("/data/module_group_sufficiency.csv").slice(1).sorted(key: r => float(r.at(6)))
// cols: 0=variant 1=label 2=category 3,4,5=per-adapter acc 6=mean_acc 7=pct_lift
#let _msbase = 0.558
#let _msceil = 0.95483
#let _msW = 10.0                       // plot width in canvas units for accuracy 0.5..1.0
#let _msxf(a) = (a - 0.5) / 0.5 * _msW
#let _suffchart = cetz.canvas({
  import cetz.draw: *
  let n = _msrows.len()
  // x gridlines + tick labels
  for t in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0) {
    let x = _msxf(t)
    line((x, -0.45), (x, n - 0.45), stroke: (paint: luma(215), dash: "dotted", thickness: 0.4pt))
    content((x, -0.75), text(size: 7pt)[#t])
  }
  // dashed base-floor and full-adapter reference lines
  line((_msxf(_msbase), -0.45), (_msxf(_msbase), n - 0.35), stroke: (paint: black, dash: "dashed", thickness: 0.6pt))
  line((_msxf(_msceil), -0.45), (_msxf(_msceil), n - 0.35), stroke: (paint: black, dash: "dashed", thickness: 0.6pt))
  content((_msxf(_msbase), n - 0.1), anchor: "south", text(size: 6.5pt)[0% of lift])
  content((_msxf(_msceil), n - 0.1), anchor: "south", text(size: 6.5pt)[100%])
  // bars, per-adapter dots, value labels, group labels
  for (i, r) in _msrows.enumerate() {
    let mean = float(r.at(6))
    let x1 = _msxf(mean)
    rect((0, i - 0.31), (x1, i + 0.31), fill: _mscolors.at(r.at(2)), stroke: 0.4pt + black)
    for c in (3, 4, 5) { circle((_msxf(float(r.at(c))), i), radius: 0.055, fill: black, stroke: none) }
    content((x1 + 0.15, i), anchor: "west",
      text(size: 7pt)[#calc.round(mean, digits: 3) (#calc.round(float(r.at(7)))%)])
    content((-0.25, i), anchor: "east", text(size: 8pt)[#r.at(1)])
  }
  content((_msW / 2, -1.15), text(size: 8pt)[Mean SST2 accuracy across 3 adapters])
})
#let _mslegenditem(k, lab) = box(baseline: 2pt, stack(dir: ltr, spacing: 3pt,
  box(width: 8pt, height: 8pt, fill: _mscolors.at(k), stroke: 0.4pt + black), text(size: 8pt)[#lab]))


#figure(
  // Scale the (natively ~13cm-wide) chart down to the current column width so it
  // never overflows a single ACL column, regardless of label lengths.
  layout(bounds => {
    let body = stack(spacing: 8pt,
      _suffchart,
      stack(dir: ltr, spacing: 10pt,
        _mslegenditem("reference", "Base / full adapter"),
        _mslegenditem("attention", "Attention (keep-only)"),
        _mslegenditem("mlp", "MLP (keep-only)"),
      ),
    )
    let s = calc.min(1.0, bounds.width / measure(body).width)
    box(scale(x: s * 100%, y: s * 100%, reflow: true, body))
  }),
  caption: [Keep-only (sufficiency) SST2 accuracy: each bar is the accuracy that ONE kept
    module-group alone reconstructs (all other LoRA components reverted to base), averaged over
    three high-accuracy adapters (per-adapter dots overlaid). Dashed lines mark the base-model floor
    ($0.558$, "0% of lift") and the full adapter ($0.955$, "100%"); the trailing "(%)" is the
    fraction of that base$arrow.r$full lift recovered.
    // @ data/module_group_sufficiency.csv; selected from 2026-08-25_kv-ablation-high-acc
    //   (its Fig. 2 sufficiency spectrum).
  ],
) <fig-module-sufficiency>

=== Causal Ablation
Informed by the LRP relevances in the previous section, we try to investigate if certain LoRA components at certain layers are more relevant than others. For that, we keep only certain LoRA components at certain layers (reverting all others to the base model), to figure out which components are sufficient to reconstruct the adapter's functionality — a sufficiency (keep-only) test in the sense of #citet("deyoung2020eraser"), and a weight-space analogue of the ablation-and-patching methodology used to localise behaviour in activations #citep("vig2020causalmediation") #citep("meng2022rome") #citep("zhang2024patching").
// FACT-CHECK: reframed "remove" -> "keep-only". The figure and every number discussed are
// keep-only (sufficiency) ablations: keep the named group, revert the rest to base. Source
// 2026-08-25_kv-ablation-high-acc also ran the complementary remove/necessity and sign-flip
// variants (e.g. sign-flipping early-MLP collapses to base while sign-flipping all attention
// barely hurts) -- not shown here.

We ablate the performance of three different SST2 LoRA adapters that all reach an accuracy of 0.95. Meanwhile the base model without any LoRA adapter reaches a performance of $0.558$. In the following we keep only certain components and measure the performance of this partly applied LoRA adapter.

As seen in @fig-module-sufficiency, keeping the most important components, MLP-Down at layer 1 to layer 11, according to the LRP map does keep most of the performance with 0.93, just slightly below 0.95, while the less important layers 12 to 16 MLP-Down stay at roughly $0.56$ accuracy -- essentially the base-model floor ($0.558$), i.e. keeping them alone recovers almost none of the adapter's performance. At the same time, keeping all attention components, which are all much less relevant according to our LRP maps, also keeps all of the performance, with roughly 0.948; even just keeping all O components keeps $0.93$ accuracy.
// FACT-CHECK: was "roughly 0.63"; the true keep-only accuracy for MLP-Down layers 12-16 is
// 0.5585 (~0% of the base->full lift). @ data/module_group_sufficiency.csv keep_mlpdown_12_16;
// verified vs source 2026-08-25_kv-ablation-high-acc (kv_ablation_results.csv). Note also that
// keep-only Q (0.559) and K/V (0.574) collapse to base, so O-proj alone carries the attention
// signal (keep_o_all 0.9295, keep_qkv_all 0.808).
Therefore, we can see some evidence that the meta model does seem to pick causally relevant components, but as shown by the O-components-only ablation, it can ignore seemingly relevant components.


#figure(
  layout(bounds => {
    let s = calc.min(1.0, bounds.width / measure(_uvchart).width)
    box(scale(x: s * 100%, y: s * 100%, reflow: true, _uvchart))
  }),
  caption: [Per-cell decision-token read magnitude $abs(v^top x)$, the norm-controlled projection of
  each cell's read direction onto the base model's activations over $400$ SST2 validation prompts.
  Orange where the decision token is the single strongest
  token category, grey where another category wins (labelled in parentheses). Bars right of the
  dashed line clear the random-direction null (95th percentile $0.037$).
  // @ data/2026-08-31_percell-regressor-uv/data/activation_readproj_summary.csv (arm B, per-cell decision vTx_abs_normed_mean; strongest category per cell); null p95 per-cell @ data/2026-08-31_percell-regressor-uv/data/activation_selectivity.csv (global p95 approx 0.037)
  ],
) <fig-uv-readproj>
=== Probing the Directions the Meta Model Reads
The causal ablations show _which_ LoRA components the meta model relies upon; they do not show _how_ it reads them. In this section we open up that read operation and ask whether the directions along which the meta model probes an adapter correspond to any direction the base model actually uses when it runs.

*Setup.* Our equivariant meta model can be written as a bilinear form over each LoRA cell, and, as @tab-compare shows, the plain bilinear regressor $#bilin8$ already predicts SST2 accuracy nearly as well as the full model. Such a model reads a cell's weight update $Delta W = B A$ by probing it with a right vector $v$ and comparing the result $Delta W v$ against a left vector $u$; that cell's contribution to the predicted accuracy is the scalar $u^top Delta W v$. The pair $(u, v)$ is thus the concrete probe the meta model applies to one adapter cell — $v$ the input (read) direction, $u$ the output (write) direction.

Decomposing the $(u, v)$ pairs of the whole-model regressor is difficult, because it bases each prediction not on one pair but on many across all components at once; doing so directly proved unsuccessful. We therefore rely on individual regressors trained on individual cells — one component at one layer at a time, e.g. Layer 13 MLP-Down. The full per-cell setup and held-out evaluation of these simplified meta models are reported in @app-percell.



We restrict the analysis to the $(u, v)$ directions that live inside the residual stream #citep("elhage2021framework"), because only those are comparable to the model's own activations. Every LLM component either reads from or writes to the residual stream #citep("elhage2021framework"): MLP-Down and Attn-O _write_ into it, so their output direction $u$ is residual-side, whereas MLP-Up, MLP-Gate and Attn-Q/K/V _read_ from it, so their input direction $v$ is residual-side. We run the base model on $400$ SST2 validation prompts, record the read projection $v^top x$ at every token, and test whether the direction fires selectively on particular tokens, which would indicate that the meta model has recovered a direction the model itself uses. As a null, each direction is compared against $128$ random unit directions in the same space, in the spirit of control tasks for probes #citep("hewitt2019controltasks") #citep("belinkov2022probing").

*Results.* In activation space the read directions fire on the decision token. Projecting each per-cell read direction $v$ onto the base model's activations over the $400$ prompts, the norm-controlled read magnitude $abs(v^top x)$ is far larger at the _decision token_ (the final `Response:` slot at which the model emits its label) than at any other token type (@fig-uv-readproj). Averaged over the $11$ per-cell directions the decision token reads $0.099$, about $3 times$ the generic tokens ($0.033$) and $6.5 times$ the random-direction null (mean $0.015$, 95th percentile $0.037$). The selectivity holds per cell: $8$ of $11$ directions exceed the null's 95th percentile on decision magnitude, and the decision token is the single strongest category in $7$ of $11$. Sentiment words ($0.040$), by contrast, read no differently from generic tokens or from the random null. The exceptions are interpretable rather than noise: the earliest attention cells, $attnKL0$ and $attnVL1$, read the BOS attention-sink #citep("xiao2024attentionsink") hardest (exactly the massive-activation residual feature that dominates layers $0$–$1$ #citep("sun2024massiveactivations")) so the decision token is not their strongest category (though $attnVL1$ still clears the null).

*Interpretation.* Together these results refine the picture from the ablations. The directions the meta model reads are legible, but as the task's _decision axis_ rather than its sentiment content: the accuracy-relevant last-layer write logit-lenses onto $mono("true")$/$mono("false")$, and the read directions fire where the classification is emitted. This is evidence that the meta model probes a causally meaningful location in the model, the position where the answer is written, rather than an arbitrary weight-statistical footprint. It falls short of proof: we show read-side alignment and selectivity, not a causal intervention, and the decision-token read is plausibly structural ("where the answer sits") rather than a semantic computation of the label.


