#import "@preview/cetz:0.3.4"
#import "@preview/pergamon:0.7.1": citep, citet
#import "../loto-scatter.typ": famscatter
#import "../lrp-maps.typ": *
// Shared native-Typst histogram panel (cetz primitives), reused by the two adapter-pool
// spread figures (@fig-loto-pool-hist and @fig-sst2-pool-hist). Draws one metric's
// distribution: blue count bars over pre-binned data, a dashed orange pool-median line,
// min/max x-range labels, and 0/peak count labels on the y-axis.
#let _histcol = rgb("#0072B2")     // adapter counts
#let _histmed = rgb("#D55E00")     // pool median
#let _histpanel(title, subtitle, lo, hi, med, counts) = cetz.canvas({
  import cetz.draw: *
  let W = 3.4
  let H = 2.1
  let nb = counts.len()
  let bw = W / nb
  let cmax = calc.max(..counts)
  let yf(c) = c / cmax * H
  let xf(x) = (x - lo) / (hi - lo) * W
  // dotted y gridlines at half and full max
  for f in (0.5, 1.0) {
    line((0, f * H), (W, f * H), stroke: (paint: luma(225), dash: "dotted", thickness: 0.4pt))
  }
  // bars
  for (i, c) in counts.enumerate() {
    rect((i * bw + 0.02, 0), ((i + 1) * bw - 0.02, yf(c)), fill: _histcol, stroke: none)
  }
  // pool median
  line((xf(med), 0), (xf(med), H), stroke: (paint: _histmed, dash: "dashed", thickness: 0.9pt))
  // frame: left + bottom axes
  line((0, 0), (0, H), stroke: 0.5pt + black)
  line((0, 0), (W, 0), stroke: 0.5pt + black)
  // y count labels (0 and max)
  content((-0.12, 0), anchor: "east", text(size: 6pt)[0])
  content((-0.12, H), anchor: "east", text(size: 6pt)[#cmax])
  // x range labels
  content((0, -0.26), anchor: "west", text(size: 6pt)[#{calc.round(lo, digits: 2)}])
  content((W, -0.26), anchor: "east", text(size: 6pt)[#{calc.round(hi, digits: 2)}])
  // title + subtitle above the panel
  content((W / 2, H + 0.55), text(size: 8pt, weight: "bold")[#title])
  content((W / 2, H + 0.22), text(size: 6pt, fill: luma(110))[#subtitle])
})

#let _histgrid(rows) = layout(bounds => {
  let panels = rows.map(r => _histpanel(..r))
  let body = stack(spacing: 12pt,
    stack(dir: ltr, spacing: 14pt, panels.at(0), panels.at(1)),
    stack(dir: ltr, spacing: 14pt, panels.at(2), panels.at(3)),
    stack(dir: ltr, spacing: 14pt, panels.at(4), panels.at(5)),
  )
  let s = calc.min(1.0, bounds.width / measure(body).width)
  box(scale(x: s * 100%, y: s * 100%, reflow: true, body))
})


= SST2 Adapter Pool: Performance Spread
The pool is deliberately 
_heterogeneous_: adapters vary in training shards, epochs, and injected label noise (up to
$0.48$). Nevertheless, every metric is heavily left-skewed: a dense mode of strong adapters near the
ceiling (median accuracy $0.91$, macro-$F_1$ $0.91$, ROC-AUC $0.97$) with a long lower tail
of degraded adapters reaching down to near-chance (accuracy $0.44$, ROC-AUC $0.47$). The
calibration metrics mirror this from the other side: NLL and Brier are right-skewed, most
mass at low (good) error with a tail of poorly calibrated adapters. Without injected label noise, the distributions of all metrics would have been even more heavily skwewed, which is why we introduced this label noise. 

#let _sst2histdata = (
  ([Accuracy], [med 0.91 · higher better], 0.441, 0.958, 0.9105,
    (20, 4, 39, 14, 9, 8, 14, 14, 14, 47, 125, 259)),
  ([Macro $F_1$], [med 0.91 · higher better], 0.306, 0.958, 0.9087,
    (38, 15, 12, 7, 11, 5, 9, 11, 20, 31, 97, 311)),
  ([ROC-AUC], [med 0.97 · higher better], 0.467, 0.990, 0.9685,
    (9, 8, 12, 13, 7, 6, 5, 7, 12, 18, 47, 423)),
  ([Mean confidence], [med 0.94 · higher better], 0.508, 0.992, 0.9355,
    (49, 49, 53, 34, 26, 17, 10, 3, 6, 15, 65, 240)),
  ([NLL], [med 0.27 · lower better], 0.134, 0.840, 0.2722,
    (130, 133, 41, 24, 31, 23, 38, 43, 31, 68, 3, 2)),
  ([Brier], [med 0.07 · lower better], 0.036, 0.306, 0.0720,
    (207, 100, 22, 26, 21, 34, 35, 26, 29, 60, 5, 2)),
)

#figure(
  _histgrid(_sst2histdata),
  caption: [Performance spread of the $567$-adapter SST2 pool ($504$ train / $63$ test)
    across the six regression-target metrics. All metrics are strongly skewed
    toward the ceiling with a long tail of noise-degraded adapters.
    // @ data/production_scores_v2.csv (all 567 rows); 12-bin counts + medians per metric.
  ],
) <fig-sst2-pool-hist>


  = Out-of-Task Pool Composition
  The out-of-task (LOTO) evaluation in @tab-loo-perdataset draws on the hold-out
  adapter pool whose per-dataset composition is given in @tab-pool-counts.

  #figure(
    table(
      columns: 2,
      align: (left, right),
      [*Dataset*], [*Trained*],
      [imdb], [$160$],  // @ data/pool_adapter_counts.csv dataset=imdb
      [rotten_tomatoes], [$160$],  // @ data/pool_adapter_counts.csv dataset=rotten_tomatoes
      [yelp_polarity], [$160$],  // @ data/pool_adapter_counts.csv dataset=yelp_polarity
      [amazon_polarity], [$160$],  // @ data/pool_adapter_counts.csv dataset=amazon_polarity
      [wiki_toxic], [$160$],  // @ data/pool_adapter_counts.csv dataset=wiki_toxic
      [toxigen], [$160$],  // @ data/pool_adapter_counts.csv dataset=toxigen
      [civil_comments], [$160$],  // @ data/pool_adapter_counts.csv dataset=civil_comments
      [wildguard_prompt_harm], [$160$],  // @ data/pool_adapter_counts.csv dataset=wildguard_prompt_harm
      [qnli], [$160$],  // @ data/pool_adapter_counts.csv dataset=qnli
      [scitail], [$160$],  // @ data/pool_adapter_counts.csv dataset=scitail
      [doc_nli], [$160$],  // @ data/pool_adapter_counts.csv dataset=doc_nli
      [vitaminc], [$160$],  // @ data/pool_adapter_counts.csv dataset=vitaminc
      [boolq], [$160$],  // @ data/pool_adapter_counts.csv dataset=boolq
      [subj], [$160$],  // @ data/pool_adapter_counts.csv dataset=subj
      [sst2], [$160$],  // @ data/pool_adapter_counts.csv dataset=sst2
      [*Total*], [$2400$],  // @ data/pool_adapter_counts.csv sum(trained_ok)
    ),
    caption: [Per-dataset adapter counts of the out-of-task hold-out pool.
      // @ data/pool_adapter_counts.csv all rows
    ],
  ) <tab-pool-counts>

  @fig-loto-pool-hist shows how the measured (held-out) performance of these
  $2400$ adapters spreads across the six metrics. Because the pool mixes $15$
  datasets across six task families (sentiment, entailment, toxicity, QA,
  subjectivity, sst2), each with its own difficulty and label balance, the
  distributions are markedly more multimodal than the single-task SST2 pool
  (@fig-sst2-pool-hist): accuracy piles up both near the balanced-binary chance
  floor ($approx 0.5$) and at a high-performing ceiling, and ROC-AUC ranges the
  full $[0.07, 0.99]$. This wide, cross-task spread is exactly the
  distribution-shift the leave-one-task-out (LOTO) evaluation stresses.

  // Native-Typst stacked histogram grid (cetz primitives): performance spread of the
  // 2400-adapter out-of-task (LOTO) hold-out pool across the six metrics, using the measured
  // (.true) columns. Bars are 12 equal-width bins over each metric's observed [min,max],
  // stacked by source dataset. The 15 datasets are colored by task family (Okabe-Ito hue)
  // and shaded within family. Rows are binned natively from data/loto_records_fam_base_l2.csv
  // at compile time (all 2400 rows) -- no frozen counts. Column order (0-indexed):
  // 3=acc.true 5=f1.true 7=auroc.true 9=brier.true 11=meanconf.true 13=nll.true.
  #let _lotorows = csv("/data/loto_records_fam_base_l2.csv").slice(1)
  #let _lotomed = luma(20)          // pool-median rule (neutral, not a family hue)
  // family base hues (Okabe-Ito, colorblind-safe)
  #let _lotofam = (
    sentiment: rgb("#0072B2"), entailment: rgb("#009E73"), toxicity: rgb("#D55E00"),
    qa: rgb("#CC79A7"), subjectivity: rgb("#56B4E9"), sst2: rgb("#E69F00"),
  )
  // datasets in family-grouped stacking order: (id, family, short legend label)
  #let _lotods = (
    ("imdb", "sentiment", "imdb"), ("rotten_tomatoes", "sentiment", "rotten"),
    ("yelp_polarity", "sentiment", "yelp"), ("amazon_polarity", "sentiment", "amazon"),
    ("qnli", "entailment", "qnli"), ("scitail", "entailment", "scitail"),
    ("doc_nli", "entailment", "docnli"), ("vitaminc", "entailment", "vitaminc"),
    ("wiki_toxic", "toxicity", "wikitox"), ("toxigen", "toxicity", "toxigen"),
    ("civil_comments", "toxicity", "civil"), ("wildguard_prompt_harm", "toxicity", "wildguard"),
    ("boolq", "qa", "boolq"), ("subj", "subjectivity", "subj"), ("sst2", "sst2", "sst2"),
  )
  // per-dataset color: family hue lightened by its rank within the family
  #let _lotocolor = {
    let sizes = (:)
    for row in _lotods { sizes.insert(row.at(1), sizes.at(row.at(1), default: 0) + 1) }
    let seen = (:)
    let out = (:)
    for row in _lotods {
      let f = row.at(1)
      let k = seen.at(f, default: 0)
      seen.insert(f, k + 1)
      let n = sizes.at(f)
      let amt = if n <= 1 { 0% } else { (k / (n - 1)) * 52% }
      out.insert(row.at(0), _lotofam.at(f).lighten(amt))
    }
    out
  }
  // native binning: returns lo, hi, median and per-bin dataset->count dicts
  #let _lotobin(col) = {
    let vals = _lotorows.map(r => float(r.at(col)))
    let lo = calc.min(..vals)
    let hi = calc.max(..vals)
    let nb = 12
    let counts = range(nb).map(_ => (:))
    for r in _lotorows {
      let b = calc.min(nb - 1, calc.floor((float(r.at(col)) - lo) / (hi - lo) * nb))
      let cur = counts.at(b)
      cur.insert(r.at(0), cur.at(r.at(0), default: 0) + 1)
      counts.at(b) = cur
    }
    let s = vals.sorted()
    let m = s.len()
    let med = if calc.rem(m, 2) == 1 { s.at(int((m - 1) / 2)) } else {
      (s.at(int(m / 2) - 1) + s.at(int(m / 2))) / 2 }
    (lo: lo, hi: hi, med: med, counts: counts)
  }
  #let _lotopanel(title, col, higher) = cetz.canvas({
    import cetz.draw: *
    let W = 3.4
    let H = 2.1
    let nb = 12
    let d = _lotobin(col)
    let lo = d.lo; let hi = d.hi; let med = d.med; let counts = d.counts
    let bintot = counts.map(c => c.values().sum(default: 0))
    let cmax = calc.max(..bintot)
    let bw = W / nb
    let yf(v) = v / cmax * H
    let xf(x) = (x - lo) / (hi - lo) * W
    for f in (0.5, 1.0) {
      line((0, f * H), (W, f * H), stroke: (paint: luma(225), dash: "dotted", thickness: 0.4pt))
    }
    // stacked bars: one segment per dataset, in family-grouped order
    for (i, c) in counts.enumerate() {
      let x0 = i * bw + 0.02
      let x1 = (i + 1) * bw - 0.02
      let acc = 0
      for row in _lotods {
        let cnt = c.at(row.at(0), default: 0)
        if cnt > 0 {
          rect((x0, yf(acc)), (x1, yf(acc + cnt)), fill: _lotocolor.at(row.at(0)), stroke: none)
          acc = acc + cnt
        }
      }
    }
    // pool median
    line((xf(med), 0), (xf(med), H), stroke: (paint: _lotomed, dash: "dashed", thickness: 0.9pt))
    // frame
    line((0, 0), (0, H), stroke: 0.5pt + black)
    line((0, 0), (W, 0), stroke: 0.5pt + black)
    // y count labels
    content((-0.12, 0), anchor: "east", text(size: 6pt)[0])
    content((-0.12, H), anchor: "east", text(size: 6pt)[#cmax])
    // x range labels
    content((0, -0.26), anchor: "west", text(size: 6pt)[#{calc.round(lo, digits: 2)}])
    content((W, -0.26), anchor: "east", text(size: 6pt)[#{calc.round(hi, digits: 2)}])
    // title + subtitle
    let dir = if higher { "higher better" } else { "lower better" }
    content((W / 2, H + 0.55), text(size: 8pt, weight: "bold")[#title])
    content((W / 2, H + 0.22), text(size: 6pt, fill: luma(110))[med #{calc.round(med, digits: 2)} · #dir])
  })
  // legend: one line per family, its datasets as shaded chips
  #let _lotochip(row) = box(baseline: 1.5pt, stack(dir: ltr, spacing: 3pt,
    box(width: 7pt, height: 7pt, fill: _lotocolor.at(row.at(0))), text(size: 6pt)[#row.at(2)]))
  #let _lotolegend = stack(dir: ttb, spacing: 3pt,
    ..("sentiment", "entailment", "toxicity", "qa", "subjectivity", "sst2").map(f =>
      stack(dir: ltr, spacing: 6pt,
        box(width: 50pt, text(size: 6.5pt, weight: "bold")[#f]),
        ..(_lotods.filter(r => r.at(1) == f)).map(_lotochip))))

  #figure(
    layout(bounds => {
      let panels = (
        _lotopanel([Accuracy], 3, true), _lotopanel([Macro $F_1$], 5, true),
        _lotopanel([ROC-AUC], 7, true), _lotopanel([Mean confidence], 11, true),
        _lotopanel([NLL], 13, false), _lotopanel([Brier], 9, false),
      )
      let body = stack(spacing: 12pt,
        stack(dir: ltr, spacing: 14pt, panels.at(0), panels.at(1)),
        stack(dir: ltr, spacing: 14pt, panels.at(2), panels.at(3)),
        stack(dir: ltr, spacing: 14pt, panels.at(4), panels.at(5)),
        _lotolegend,
      )
      let s = calc.min(1.0, bounds.width / measure(body).width)
      box(scale(x: s * 100%, y: s * 100%, reflow: true, body))
    }),
    caption: [Performance spread of the $2400$-adapter out-of-task (LOTO) hold-out pool
      across the six metrics, using each adapter's measured held-out score. Higher is
      better for accuracy, macro-$F_1$, ROC-AUC and mean confidence; lower for NLL and Brier.

    ],
  ) <fig-loto-pool-hist>

  // TODO: the LRP Maps section (see @in-task-eval interpretability) states the
  // seed noise of the LRP maps is "analysed in the appendix", but no such
  // figure/analysis exists yet. Add the seed-noise figure here and reference it
  // from the "=== LRP Maps" paragraph, or drop that sentence.

= Adapter Pool Tasks and Hyperparameters

The adapter pool is trained across $15$ binary classification tasks, grouped into four families
(sentiment, toxicity, entailment, and two other tasks) and summarised in @tab-tasks.

#figure(
  caption: [The 15 binary classification tasks used to train the adapter pools, grouped into four
    families (sentiment, toxicity, entailment, and two other tasks).],
  table(
    columns: (auto, auto),
    align: (left, left),
    table.header(
      [*Task*], [*Type*],
    ),
    table.cell(colspan: 2, fill: luma(238))[*Sentiment*],
    [`sst2` #citep("socher2013sst2")],            [Sentiment],
    [`imdb` #citep("maas2011imdb")],            [Sentiment],
    [`rotten_tomatoes` #citep("pang2005rottentomatoes")], [Sentiment],
    [`yelp_polarity` #citep("zhang2015charcnn")],   [Sentiment],
    [`amazon_polarity` #citep("mcauley2013amazon") #citep("zhang2015charcnn")], [Sentiment],
    table.cell(colspan: 2, fill: luma(238))[*Toxicity*],
    [`wiki_toxic` #citep("jigsaw2018toxic")],      [Toxicity],
    [`toxigen` #citep("hartvigsen2022toxigen")],         [Toxicity],
    [`civil_comments` #citep("borkan2019civilcomments")],  [Toxicity],
    [`wildguard` #citep("han2024wildguard")],       [Toxicity],
    table.cell(colspan: 2, fill: luma(238))[*Entailment*],
    [`qnli` #citep("wang2018glue") #citep("rajpurkar2016squad")],            [Entailment],
    [`scitail` #citep("khot2018scitail")],         [Entailment],
    [`doc_nli` #citep("yin2021docnli")],         [Entailment],
    [`vitaminc` #citep("schuster2021vitaminc")],        [Entailment],
    table.cell(colspan: 2, fill: luma(238))[*Other*],
    [`boolq` #citep("clark2019boolq")], [QA],
    [`subj` #citep("pang2004subj")],  [Subjectivity],
  ),
) <tab-tasks>

== LoRA Hyperparameter Sampling
LoRA adapter training requires selecting numerous hyperparameters that impact the training. To avoid confounding between performance and hyperparameter configuration, we randomly sample the training hyperparameters from fixed ranges and later analyse whether performance can be predicted from the sampled configuration itself. Additionally, we inject label noise (flipping random subsets of training labels) to increase the accuracy spread across adapters; otherwise most adapters trained on SST2 would sit at either perfect or near-baseline performance. The sampled ranges are given in @tab-hparams.

#figure(
  caption: [Sampled hyperparameter ranges for the LoRA adapters; each value is drawn independently per adapter.],
  table(
    columns: (auto, auto, auto),
    align: (left, left, left),
    table.header(
      [*Hyperparameter*], [*Distribution*], [*Range*],
    ),
    [Learning rate],   [Linear-uniform], [$[5 times 10^(-5), 3 times 10^(-4)]$],
    [Epochs],          [Uniform choice], [${1, 2, 3}$],
    [Effective batch], [Fixed],          [$32$],
    [LoRA dropout],    [Uniform],        [$[0, 0.1]$],
    [LoRA alpha],      [Uniform choice], [${16, 32}$],
  ),
) <tab-hparams>
// TODO: quantify the label-noise injection rate (and data-shard count, if used) -- these are
//       referenced in the prose as the driver of accuracy spread but are not given a range here.

= Geometric Feature Definitions <appendix-geometric-features>

The scalar weight-geometry features used by the ridge baselines fall into two groups: the
_intrinsic spectral features_ (@tab-metrics-intrinsic), derived solely from the singular values of
the LoRA update $Delta W$, and the _base-relative features_ (@tab-metrics-baserel), which
characterise $Delta W$ against the frozen base weight $W_0$.

#figure(
  caption: [Intrinsic spectral features: scalars derived from the singular values $sigma$ of the LoRA
    update $Delta W$ alone, used as features for our ridge regressor.],
  text(size: 8pt, table(
    columns: (auto, 1fr),
    align: (left, left),
    inset: (x: 4pt, y: 2.5pt),
    table.header([*Feature*], [*Description*]),
    [Spectral norm $sigma_1$], [Largest singular value of $Delta W$.],
    [Nuclear norm $sum_i sigma_i$], [Sum of the singular values.],
    [Stable rank $||Delta W||_F^2 slash sigma_1^2$], [How spread the update's energy is across the spectrum #citep("rudelson2007stablerank").],
    [Effective (participation) rank], [Entropy-based soft rank of the normalised spectrum #citep("roy2007effectiverank").],
    [Spectral entropy], [Shannon entropy of the normalised singular values.],
    [Log–log decay slope], [Slope of the singular values on a log–log axis (tail heaviness).],
    [Spectral kurtosis], [Tailedness of the singular-value distribution.],
    [Spectral skew], [Asymmetry of the singular-value distribution.],
  )),
) <tab-metrics-intrinsic>

#figure(
  caption: [Base-relative features: scalars derived from the LoRA update $Delta W$ relative to the base weight, used as features for our ridge regressor.],
  text(size: 8pt, table(
    columns: (auto, 1fr),
    align: (left, left),
    inset: (x: 4pt, y: 2.5pt),
    table.header([*Feature*], [*Description*]),
    [Amplification], [$||U_k^top Delta W V_k|| slash ||Delta W||$: fraction of the update's energy lying inside $W_0$'s dominant subspace #citep("hu2022lora").],
    [Novel-direction fraction], [$1 - ||P_(U_k) Delta W|| slash ||Delta W||$: fraction of the update pointing in directions new to $W_0$ #citep("shuttleworth2025illusion").],
    [Principal angle (mean $cos^2$)], [Average alignment between $Delta W$'s column space and $U_k$ #citep("bjorck1973principalangles").],
    [Principal angle (smallest)], [Closest angle between $Delta W$'s column space and $U_k$.],
    [Spectral displacement $s_1$–$s_4$], [Four terms measuring how $Delta W$ shifts the base singular spectrum.],
    [Base-relative norm $||Delta W||_F slash ||W_0||_F$], [Overall size of the update relative to the base weight.],
  )),
) <tab-metrics-baserel>

= Chat Templates of the Adapter Pools

As previously established, all tasks are binary tasks. To avoid confounding between the arbitrarily picked labels and their corresponding tasks, we standardise all chat templates across all tasks to use uniform "true" and "false" labels as prediction targets. Each task uses just one chat template. Both measures reduce the difficulty of the task, as we require no generalisation across output labels or chat templates from the meta model. The chat templates are descriptive: each contains both the task and a description of how to respond, so even an untrained model could plausibly solve the task. No task shares its chat template with any other task.

// The system prompt each dataset ships with, all rendered under the TRUE_FALSE_V1
// verbalizer. Source of truth (quoted verbatim at code commit ec5d9e3, base model
// meta-llama/Llama-3.2-1B, LoRA rank-16):
// data/2026-08-31_chat-templates-adapter-pools/report.typ.

Every dataset renders its prompt with the same generic layout: the system prompt, a blank
line, one #raw("Field: value") line per input field, and finally the response token with
the label as the completion.

#figure(
  box(
    width: 100%,
    stroke: 0.5pt + luma(180),
    inset: 8pt,
    align(left, ```
    <system prompt>

    <FieldA>: <text>
    [<FieldB>: <text>]
    Response: <label>
    ```),
  ),
  caption: [Generic chat template shared by every dataset: the system prompt, a
    blank line, one #raw("Field: value") line per input field, and the response
    token with the label as the completion.],
) <fig-chat-template>

The input fields per dataset are:

#figure(
  table(
    columns: 2,
    align: (left, left),
    table.header([*Input fields*], [*Datasets*]),
    [#raw("Review:")], [`imdb`, `rotten_tomatoes`, `yelp_polarity`],
    [#raw("Title:") #raw("Review:")], [`amazon_polarity`],
    [#raw("Comment:")], [`wiki_toxic`, `civil_comments`],
    [#raw("Statement:")], [`toxigen`],
    [#raw("Request:")], [`wildguard`],
    [#raw("Question:") #raw("Sentence:")], [`qnli`],
    [#raw("Premise:") #raw("Hypothesis:")], [`scitail`],
    [#raw("Document:") #raw("Hypothesis:")], [`doc_nli`],
    [#raw("Evidence:") #raw("Claim:")], [`vitaminc`],
    [#raw("Passage:") #raw("Question:")], [`boolq`],
    [#raw("Sentence:")], [`subj`, `sst2`],
  ),
  caption: [Input fields used in the chat template for each dataset.],
) <tab-input-fields>

== Two full worked examples — SST2 and `wiki_toxic`:

#figure(
  box(
    stroke: 0.5pt + luma(180),
    inset: 8pt,
    ```
    Given a sentence from a movie review, your task is to classify the sentiment as
    either true or false. Consider the overall tone and word choice. Your response
    should clearly indicate whether the sentiment is true or false.

    Sentence: <sentence>
    Response: <label>
    ```,
  ),
  caption: [Worked example of the chat template for the SST2 sentiment prompt.],
) <fig-worked-example-sst2>

#figure(
  box(
    stroke: 0.5pt + luma(180),
    inset: 8pt,
    ```
    Given a comment from a Wikipedia talk page, your task is to classify it as either
    toxic or non-toxic. A toxic comment is rude, disrespectful, or insulting and would
    make someone leave the discussion. Your response should clearly indicate whether
    the comment is toxic or non-toxic.

    Comment: <comment>
    Response: <label>
    ```,
  ),
  caption: [Worked example of the chat template for the #raw("wiki_toxic") toxicity prompt.],
) <fig-worked-example-wikitoxic>

== System Prompts of all Datasets:

/ #raw("imdb"): Given a movie review, your task is to classify the sentiment of the review as either true or false. Consider the overall tone, the choice of words, and the context in which sentiments are expressed. Your response should clearly indicate whether the sentiment of the review is true or false.
/ #raw("rotten_tomatoes"): Given a movie review, your task is to classify the sentiment of the review as either true or false. Consider the overall tone, the choice of words, and the context in which sentiments are expressed. Your response should clearly indicate whether the sentiment of the review is true or false.
/ #raw("yelp_polarity"): Given a Yelp review, your task is to classify the sentiment of the review as either true or false. Consider the overall tone, the choice of words, and the context in which sentiments are expressed. Your response should clearly indicate whether the sentiment of the review is true or false.
/ #raw("amazon_polarity"): Given a product review, your task is to classify the sentiment as either positive or negative. Consider the overall tone, word choice, and context. Your response should clearly indicate whether the sentiment is positive or negative.
/ #raw("wiki_toxic"): Given a comment from a Wikipedia talk page, your task is to classify it as either toxic or non-toxic. A toxic comment is rude, disrespectful, or insulting and would make someone leave the discussion. Your response should clearly indicate whether the comment is toxic or non-toxic.
/ #raw("toxigen"): Given a statement, your task is to classify it as either toxic or benign. A toxic statement expresses or implies hatred, prejudice, or harm toward a group of people, even when phrased subtly. Your response should clearly indicate whether the statement is toxic or benign.
/ #raw("civil_comments"): Given an online comment, your task is to classify it as either toxic or non-toxic. A toxic comment is rude, disrespectful, or insulting and would make someone leave a discussion. Your response should clearly indicate whether the comment is toxic or non-toxic.
/ #raw("wildguard_prompt_harm"): Given a request that a user has sent to an AI assistant, your task is to classify whether the request is harmful or unharmful. A harmful request seeks content that could cause real-world harm, such as instructions for violence, illegal activity, or dangerous material. Your response should clearly indicate whether the request is harmful or unharmful.
/ #raw("qnli"): You are given a question and a sentence. Your task is to determine whether the sentence contains the answer to the question. Respond with exactly one label: true or false.
/ #raw("scitail"): You are given a premise sentence and a hypothesis sentence drawn from a science exam. Your task is to determine whether the hypothesis is entailed by the premise. Your response should clearly indicate whether the hypothesis is true or false.
/ #raw("doc_nli"): You are given a document and a hypothesis sentence. Your task is to determine whether the hypothesis is true by the document. Your response should clearly indicate whether the hypothesis is true or false.
/ #raw("vitaminc"): You are given a piece of evidence and a claim. Your task is to determine whether the evidence supports or refutes the claim. Your response should be exactly one word: true or false.
/ #raw("boolq"): You are given a passage and a question. Your task is to determine whether the answer to the question is true or false based on the passage. Respond with exactly one word: true or false.
/ #raw("subj"): Given a sentence, your task is to classify it as either objective or subjective. An objective sentence states a fact or describes a plot event without personal opinion, while a subjective sentence expresses a personal opinion, judgement, or feeling. Your response should clearly indicate whether the sentence is objective or subjective.
/ #raw("sst2"): Given a sentence from a movie review, your task is to classify the sentiment as either true or false. Consider the overall tone and word choice. Your response should clearly indicate whether the sentiment is true or false.

= Single-Head Complexity Ladder
To evaluate the benefit of increasing the equivariant width, we evaluate a series of regression models using different configurations.
We train each model on the same SST2 performance regression tasks as in @in-task-eval and just on the accuracay (single-head).  We iterate model settings using the widths $d in {1, 4, 6, 8, 12, 16}$ . To ensure significants, we train and evaluate three seeds per model configruations.

As seen in the data, the step from $d=1$ to $d=4$ is by far the most relevant across both the $R^2$ and $p$ metric, as the $R^2$ score increases from ($approx -0.06$ to $$. Meanwhile the spearmann $tau$ increases from $$ to $$. Additionally, it can be observed that increasing $d$ to $8$ does improve the $R^2$ score to approx $$.


// Native-Typst complexity-ladder chart (cetz primitives). Two stacked panels share a
// log2(d) x-axis: held-out R^2 (top) and Spearman rho (bottom) of the accuracy-only
// single-equivariant-layer SST2 regressor across width d in {1,4,6,8,12,16}. Points are
// the mean over 3 seeds (42/43/44), whiskers +-1 sample std, dashed orange = parent
// 6-head deep anchor base-l2. Data frozen in data/complexity_ladder_3seed.csv (a 3-seed
// re-aggregation of data/ladder/data/acc_ladder_per_seed.csv); anchors from
// data/ladder/data/parent_anchors.csv (winner_base_l2).
#let _clrows = csv("/data/complexity_ladder_3seed.csv").slice(1)
// cols: 0=arch 1=d 2=n_seeds 3=r2_mean 4=r2_std 5=spearman_mean 6=spearman_std
#let _clline = rgb("#009E73")     // accuracy-only ladder
#let _clanchor = rgb("#D55E00")   // parent 6-head deep anchor base-l2
#let _clW = 8.0                   // panel width in canvas units
#let _clxf(d) = calc.log(d, base: 2) / 4.0 * _clW   // log2(d)/4 -> 0..W for d in 1..16

#let _clpanel(ycol, ecol, ymin, ymax, panelh, ticks, title, showx) = cetz.canvas({
  import cetz.draw: *
  let yf(v) = (v - ymin) / (ymax - ymin) * panelh
  // dotted y gridlines + tick labels
  for t in ticks {
    let y = yf(t)
    line((0, y), (_clW, y), stroke: (paint: luma(222), dash: "dotted", thickness: 0.4pt))
    content((-0.28, y), anchor: "east", text(size: 7pt)[#t])
  }
  // solid zero line when it falls inside the range
  if ymin < 0 and ymax > 0 {
    line((0, yf(0)), (_clW, yf(0)), stroke: (paint: luma(150), thickness: 0.5pt))
  }
  // frame: left + bottom axes
  line((0, 0), (0, panelh), stroke: 0.5pt + black)
  line((0, 0), (_clW, 0), stroke: 0.5pt + black)
  // ladder line through the per-rung means
  let pts = _clrows.map(r => (_clxf(float(r.at(1))), float(r.at(ycol)), float(r.at(ecol))))
  line(..pts.map(p => (p.at(0), yf(p.at(1)))), stroke: (paint: _clline, thickness: 1pt))
  // per-rung whiskers (+-1 std) + mean markers
  for p in pts {
    let x = p.at(0); let m = p.at(1); let e = p.at(2)
    line((x, yf(m - e)), (x, yf(m + e)), stroke: (paint: _clline, thickness: 0.7pt))
    line((x - 0.1, yf(m - e)), (x + 0.1, yf(m - e)), stroke: (paint: _clline, thickness: 0.7pt))
    line((x - 0.1, yf(m + e)), (x + 0.1, yf(m + e)), stroke: (paint: _clline, thickness: 0.7pt))
    circle((x, yf(m)), radius: 0.09, fill: _clline, stroke: none)
  }
  // panel title (top-left)
  content((0.15, panelh - 0.25), anchor: "west", text(size: 8pt, weight: "bold")[#title])
  // x tick labels + axis title on the bottom panel only
  if showx {
    for r in _clrows {
      let x = _clxf(float(r.at(1)))
      content((x, -0.5), text(size: 7pt)[#r.at(1)])
    }
    content((_clW / 2, -1.05), text(size: 8pt)[equivariant width $d$ ($log_2$ axis)])
  }
})

#let _cllegenditem(sym, lab) = box(baseline: 2pt, stack(dir: ltr, spacing: 4pt, sym, text(size: 8pt)[#lab]))

#figure(
  layout(bounds => {
    let body = stack(spacing: 6pt,
      _clpanel(3, 4, -0.2, 0.9, 4.0, (0.0, 0.2, 0.4, 0.6, 0.8),
        [Held-out $R^2$], false),
      _clpanel(5, 6, 0.2, 1.0, 3.0, (0.2, 0.4, 0.6, 0.8, 1.0),
        [Spearman $rho$], true),
      stack(dir: ltr, spacing: 12pt,
        _cllegenditem(box(width: 14pt, height: 2pt, stack(dir: ltr, spacing: 0pt,
          line(length: 14pt, stroke: 1pt + _clline))), "accuracy-only ladder (mean ± std, 3 seeds)"),
      ),
    )
    let s = calc.min(1.0, bounds.width / measure(body).width)
    box(scale(x: s * 100%, y: s * 100%, reflow: true, body))
  }),
  caption: [Single-head complexity ladder: held-out SST2 calibration ($R^2$, top) and
    ranking (Spearman $rho$, bottom) of the accuracy-only single-equivariant-layer
    regressor versus equivariant width $d$ (log scale).
  ],
) <fig-complexity-ladder>



= Multi-Head Meta Model compared to Single-Head Meta Model
In the report, we mostly focused on multi-head meta models, so regression models trained to predict all six SST2 metric targets at once. To find out if a single-head meta model would outperform or would be outperformed by those multi-head meta models, we compare both types of models. We hold all other parameters equal otherwise, so the same dense head, the SST2 adapter pool, same seeds.





@fig-head-compare shows the seed-matched held-out calibration $R^2$. The multi-head model
calibrates better at all three widths; the paired gap is largest where $d = 4$ with $Delta R^2 = -0.089$ and settles to $-0.028 slash -0.041$ once width
saturates. The effect is $R^2$-specific: the matched
ranking gap is only $Delta rho approx -0.006 "to" -0.008$, so accuracy alone already ranks held-out adapters and the five auxiliary heads act
as a small, consistent calibration regulariser that never hurts.

// Grouped bar chart (cetz primitives): seed-matched held-out R^2 of the multi-head
// (six-target) vs single-head (accuracy-only) regressor at equivariant width d in {4,8,16},
// means over the 3 common seeds 42/43/44, whiskers +-1 sample std. Data frozen from
// data/2026-08-31_gl-regressor-head-count-fair-comparison/data/paired_deltas.csv
// (re-aggregated to seeds 42/43/44 only); base-l2 ceiling R^2=0.825 from that report.
#let _hcmulti = rgb("#0072B2")   // multi-head (six-target)
#let _hcacc   = rgb("#D55E00")   // single-head (accuracy-only)
#let _hcW = 8.0                  // canvas width in units
#let _hcH = 5.0                  // canvas height in units
#let _hcymin = 0.70
#let _hcymax = 0.90
// rows: (d, multi_mean, multi_std, acc_mean, acc_std)
#let _hcdata = (
  ("4",  0.8276, 0.0399, 0.7387, 0.0264),
  ("8",  0.8415, 0.0226, 0.8132, 0.0299),
  ("16", 0.8556, 0.0200, 0.8145, 0.0127),
)

#let _hclegenditem(col, lab) = box(baseline: 2pt, stack(dir: ltr, spacing: 4pt,
  box(width: 10pt, height: 8pt, fill: col), text(size: 8pt)[#lab]))

#figure(
  cetz.canvas({
    import cetz.draw: *
    let yf(v) = (v - _hcymin) / (_hcymax - _hcymin) * _hcH
    let ticks = (0.70, 0.75, 0.80, 0.85, 0.90)
    // dotted y gridlines + tick labels
    for t in ticks {
      let y = yf(t)
      line((0, y), (_hcW, y), stroke: (paint: luma(222), dash: "dotted", thickness: 0.4pt))
      content((-0.28, y), anchor: "east", text(size: 7pt)[#t])
    }
    // parent deep six-head ceiling (base-l2)
    let ceil = 0.8252
    line((0, yf(ceil)), (_hcW, yf(ceil)),
      stroke: (paint: luma(110), dash: "dashed", thickness: 0.7pt))
    content((_hcW, yf(ceil) + 0.1), anchor: "south-east",
      text(size: 6.5pt, fill: luma(110))[w128-l2 ceiling = 0.825])
    // frame: left + bottom axes
    line((0, 0), (0, _hcH), stroke: 0.5pt + black)
    line((0, 0), (_hcW, 0), stroke: 0.5pt + black)
    // grouped bars
    let n = _hcdata.len()
    let slot = _hcW / n
    let bw = 0.62      // bar width
    let gap = 0.18     // gap between the two bars in a group
    for (i, row) in _hcdata.enumerate() {
      let cx = slot * (i + 0.5)
      let mx = cx - gap / 2 - bw
      let ax = cx + gap / 2
      let mm = row.at(1); let ms = row.at(2)
      let am = row.at(3); let as_ = row.at(4)
      // bars
      rect((mx, 0), (mx + bw, yf(mm)), fill: _hcmulti, stroke: none)
      rect((ax, 0), (ax + bw, yf(am)), fill: _hcacc, stroke: none)
      // whiskers (+-1 std)
      for (bcx, m, e) in ((mx + bw / 2, mm, ms), (ax + bw / 2, am, as_)) {
        line((bcx, yf(m - e)), (bcx, yf(m + e)), stroke: (paint: luma(60), thickness: 0.7pt))
        line((bcx - 0.09, yf(m - e)), (bcx + 0.09, yf(m - e)), stroke: (paint: luma(60), thickness: 0.7pt))
        line((bcx - 0.09, yf(m + e)), (bcx + 0.09, yf(m + e)), stroke: (paint: luma(60), thickness: 0.7pt))
      }
      // value labels above each bar's whisker
      content((mx + bw / 2, yf(mm + ms) + 0.22), text(size: 6.5pt, fill: _hcmulti)[#{calc.round(mm, digits: 3)}])
      content((ax + bw / 2, yf(am + as_) + 0.22), text(size: 6.5pt, fill: _hcacc)[#{calc.round(am, digits: 3)}])
      // x group label
      content((cx, -0.42), text(size: 8.5pt)[$d = #row.at(0)$])
    }
    // y axis title
    content((-0.95, _hcH / 2), angle: 90deg, text(size: 8pt)[held-out $R^2$])
    // x axis title
    content((_hcW / 2, -0.95), text(size: 8pt)[equivariant width $d$])
  }),
  caption: [Multi-head versus single-head meta model: seed-matched held-out SST2
    calibration ($R^2$) at equivariant widths $d in {4, 8, 16}$. Bars are the mean over the
    three common seeds ($42 slash 43 slash 44$), whiskers $plus.minus 1$ standard deviation;
    blue is the multi-head regressor (all six metric targets), orange the single-head model
    (accuracy alone). The dashed line is the parent's $d = 128$ six-head ceiling
    #raw("w128-l2") ($R^2 = 0.825$). The multi-head model wins at every width, by a
    width-shrinking margin ($Delta R^2 = -0.089 slash -0.028 slash -0.041$).
    // @ data/2026-08-31_gl-regressor-head-count-fair-comparison/data/paired_deltas.csv
    //   (seeds 42/43/44 only); ceiling from that report's base-l2 anchor.
  ],
) <fig-head-compare>



= Per-Cell Bilinear Meta Models <app-percell>

// Per-cell (module, layer) LoRA site names, mirroring the definitions in sections/results.typ.
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

The UV analysis in the results section relies on bilinear
meta models trained on a _single_ LoRA cell (module, layer) rather than on all cells
jointly. Here we enumerate those cells and report how well each single cell predicts held-out SST2
accuracy on its own.

*Cells.* Eleven cells are studied on the $16$-layer base model, in three groups. Six _important_
cells are the top accuracy-relevant site per module, taken from the whole-model regressor's own
per-cell load ranking: $attnOL11$, $attnQL10$, $gateL13$, $downL12$, $upL11$, $attnVL1$. Four
_control_ cells are early / low-relevance sites: $attnKL0$, $gateL0$, $upL0$, $downL1$. One
_last-layer_ cell, $downL15$, is included for the logit lens #citep("nostalgebraist2020logitlens") #citep("belrose2023tunedlens"). The residual-side (logit-lensable)
direction is the write direction $u$ for the MLP-down and Attn-O writers and the read direction $v$
for the Attn-Q/K/V, MLP-gate and MLP-up readers.

*Setup.* Every cell is fit at equivariant width $d = 8$ with L2-normalised
features, matching the whole-model regressor; the meta model is direction-only, with a per-adapter
template whose reported $(u, v)$ is a sign-aligned mean over the $63$ test adapters. All regressors
use a single seed and are trained on the same $504$-train / $63$-test SST2 pool as the whole-model
experiments, targeting the SST2-test likelihood accuracy.

*Results.* @tab-percell-metrics gives, per cell, the held-out accuracy prediction
($R^2$ and Spearman $rho$). It predicts held-out accuracy reasonably well (mean $R^2 = 0.25$, $rho = 0.75$), peaking at
$downL12$ ($R^2 = 0.68$, $rho = 0.90$). Predictiveness is fairly distributed rather than concentrated
in the "important" cells, e.g the control cell $downL1$ reaches $R^2 = 0.52$, comparable to the
important $upL11$ ($0.50$).

#figure(
  text(8pt, table(
    columns: (2.1cm, 1.5cm, 1.2cm, 1.2cm),
    align: (left, left, right, right),
    table.header(
      [*Cell*], [*Group*], [*$R^2$*], [*$rho$*],
    ),
    [$attnOL11$], [important], [$0.28$], [$0.83$],
    [$attnQL10$], [important], [$0.17$], [$0.75$],
    [$gateL13$],  [important], [$0.17$], [$0.78$],
    [$downL12$],  [important], [$0.68$], [$0.90$],
    [$upL11$],    [important], [$0.50$], [$0.84$],
    [$attnVL1$],  [important], [$0.16$], [$0.71$],
    [$attnKL0$],  [control],   [$-0.08$],[$0.52$],
    [$gateL0$],   [control],   [$-0.15$],[$0.61$],
    [$upL0$],     [control],   [$0.11$], [$0.70$],
    [$downL1$],   [control],   [$0.52$], [$0.78$],
    [$downL15$],  [last-layer],[$0.33$], [$0.83$],
  )),
  caption: [Per-cell L2-normalised bilinear meta models: held-out SST2 accuracy prediction ($R^2$ and
  Spearman $rho$) over the $63$ test adapters. Templates stay spread ($sigma_0$-share $approx 0.71$)
  yet predict held-out accuracy well almost everywhere.
  // @ data/2026-08-31_percell-regressor-uv/data/cell_regression_metrics.csv (arm B: cols r2, spearman per cell); sigma0_share in data/2026-08-31_percell-regressor-uv/data/uv_summary.csv
  ],
) <tab-percell-metrics>



// ============================================================================
// SUPPLEMENTARY FIGURES
// FLAG: these figures were moved out of the main flow. This "=" heading only parks them at the end
//       of THIS file -- it is NOT the document's real appendix. If main.typ has a central
//       #appendix[...] block (it does -- it holds tab-pool-counts), move these figure blocks into
//       that block so they render as true appendix material rather than a trailing Results section.
// ============================================================================
= Supplementary Figures


#let basel2 = math.op("w128-l2")
#let base0 = math.op("w128")
#let bilin8 = math.op("w8-bilin-nohead-l2")
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

== Out-of-Task (LOTO) Scatter Plots
The sentiment-family LOTO scatter is shown in the main text (@fig-loto-sentiment); the remaining
two families are given here.

#figure(famscatter("toxicity", size: (5.6, 4.6)),
  caption: [Toxicity family — $#basel2$ LOTO predicted vs. true SST2 accuracy, per dataset.]
) <fig-loto-toxicity>

#figure(famscatter("entailment", size: (5.6, 4.6)),
  caption: [Entailment family — $#basel2$ LOTO predicted vs. true SST2 accuracy, per dataset.]
) <fig-loto-entailment>

== LRP Map Decomposition
The PCA decomposition and accuracy-cluster maps supporting the LRP analysis (@sec:lrp) are
given here.

#figure(
  stack(spacing: 12pt,
    _hmpanel(_tg, _blue, _red, [(a) PCA mean — the assumed centre], cell: 8pt, layeraxis: true),
    _hmpanel(_pc1, _pur, _org, [(b) PC1 loading — the accuracy axis], cell: 8pt, layeraxis: true),
  ),
  caption: [PCA decomposition of the #w8 accuracy-head LRP maps across all tested adapters..
    *(a)* the PCA mean (the shared template PCA subtracts). *(b)* the PC1 loading: the high-vs-low
    accuracy axis: orange ($+$) is the high-accuracy direction, purple ($-$) is the low-accuracay direction
  ],
) <fig-lrp-pca>

#figure(
  stack(spacing: 10pt,
    align(center, stack(spacing: 3pt,
      text(size: 8pt, weight: "bold")[(a) cluster cohesion (mean pairwise cosine)], _cohesioncanvas)),
    align(center, stack(spacing: 5pt,
      text(size: 8pt, weight: "bold")[(b) three typical high-acc maps · three distinctive low-acc maps],
      grid(columns: 3, column-gutter: 8pt, row-gutter: 8pt, align: center + bottom,
        ..for (aid, acc, hi) in _exadapters {
          let g = _parsegrid(_exrows.filter(r => r.at(0) == aid), 4, 5, 6)
          (stack(spacing: 3pt,
            text(size: 6pt, fill: if hi { rgb("#1b7837") } else { rgb("#b30000") })[
              #(if hi { "typical hi" } else { "distinct lo" }) \ acc #acc],
            _heatmap(g, _blue, _red, cell: 4pt, labels: false, layeraxis: true)),)
        }))),
  ),
  caption: [Accuracy clusters. *(a)* mean pairwise map cosine within the high-accuracy tercile
    ($0.98$), within the low-accuracy tercile ($0.61$), and across the two ($0.48$). *(b)* the three
    most-typical high-accuracy maps beside the three
    most-distinctive low-accuracy maps.
    // @ data/cluster_cohesion.csv, data/cluster_example_grids.csv; recreated from
    //   2026-08-25_gl-regressor-lrp-acc fig. 9.
  ],
) <fig-lrp-clusters>
