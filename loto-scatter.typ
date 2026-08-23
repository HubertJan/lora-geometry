// Native-Typst per-family LOTO scatter of base_l2 predictions (cetz; reads the frozen
// out-of-task records copied from the sibling ood-nonsentiment-families experiment).
#import "@preview/cetz:0.3.4"
#import "@preview/cetz-plot:0.1.1": plot

#let _recs = csv("data/loto_records_fam_base_l2.csv").slice(1)  // cols: 1=dataset 3=acc.true 4=acc.pred
#let _famsets = (
  sentiment: ("imdb", "rotten_tomatoes", "yelp_polarity", "amazon_polarity"),
  toxicity: ("wiki_toxic", "toxigen", "civil_comments", "wildguard_prompt_harm"),
  entailment: ("qnli", "scitail", "doc_nli", "vitaminc"),
)
#let _cols = (blue, red, green, orange)
#let famscatter(fam, size: (4.6, 4.2), show-y: true) = {
  let vals = ()
  for ds in _famsets.at(fam) {
    for r in _recs.filter(r => r.at(1) == ds) { vals.push(float(r.at(3))); vals.push(float(r.at(4))) }
  }
  let lo = calc.min(..vals)
  let hi = calc.max(..vals)
  text(7pt, cetz.canvas({
    plot.plot(size: size, x-label: "true SST2 accuracy",
      y-label: if show-y { "predicted SST2 accuracy" } else { none },
      x-tick-step: 0.2, y-tick-step: 0.2,
      legend: "inner-north-west", legend-style: (item: (spacing: .12), padding: .1),
      {
      for (k, ds) in _famsets.at(fam).enumerate() {
        let pts = _recs.filter(r => r.at(1) == ds).map(r => (float(r.at(3)), float(r.at(4))))
        plot.add(pts, mark: "o", mark-size: .06, mark-style: (fill: _cols.at(k), stroke: none),
          style: (stroke: none), label: ds)
      }
      plot.add(((lo, lo), (hi, hi)), style: (stroke: (paint: black, dash: "dashed")))
    })
  }))
}
