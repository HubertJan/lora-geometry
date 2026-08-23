// ============================================================================
// Native-Typst LRP map figures, recreated from the sibling report
// 2026-08-25_gl-regressor-lrp-acc (Acc-head LRP of the w8 regressor; its figs 7/8/9).
// A map is 7 modules × 16 LLM layers of *signed* relevance drawn as coloured cells
// (RdBu_r for relevance, PuOr_r for the PC1 loading); scree/histogram/bars via cetz.
// Data CSVs were copied verbatim from that report's data/ into this folder's data/.
// ============================================================================
#import "@preview/cetz:0.3.4"
#import "@preview/cetz-plot:0.1.1": plot

#let _MODS = ("Q", "K", "V", "O", "gate", "up", "down")
#let _white = (255, 255, 255)
#let _red = (178, 24, 43)      // RdBu_r positive
#let _blue = (33, 102, 172)    // RdBu_r negative
#let _org = (230, 97, 1)       // PuOr_r positive (high-accuracy direction)
#let _pur = (94, 60, 153)      // PuOr_r negative
#let _mkrgb(c) = rgb(c.at(0), c.at(1), c.at(2))
#let _ilerp(a, b, t) = _mkrgb((0, 1, 2).map(i => int(calc.round(a.at(i) + (b.at(i) - a.at(i)) * t))))
#let _diverge(v, vmax, neg, pos) = {
  let t = calc.max(-1.0, calc.min(1.0, v / vmax))
  if t >= 0 { _ilerp(_white, pos, t) } else { _ilerp(_white, neg, -t) }
}
#let _parsegrid(rows, mi, li, vi) = {
  let m = (:); let vmax = 0.0
  for r in rows {
    let v = float(r.at(vi))
    m.insert(r.at(mi) + "|" + r.at(li), v)
    vmax = calc.max(vmax, calc.abs(v))
  }
  (map: m, vmax: vmax)
}
#let _heatmap(g, neg, pos, cell: 7pt, labels: true, layeraxis: false) = {
  let cells = ()
  for mod in _MODS {
    if labels { cells.push(align(right + horizon, text(size: 5.5pt, mod + h(1.5pt)))) }
    for l in range(16) {
      let v = g.map.at(mod + "|" + str(l), default: 0.0)
      cells.push(box(width: cell, height: cell, fill: _diverge(v, g.vmax, neg, pos),
        stroke: 0.15pt + luma(235)))
    }
  }
  if layeraxis {
    // bottom axis: one LLM-layer index (0..15) per column, rotated to fit narrow cells
    if labels { cells.push([]) }
    for l in range(16) {
      cells.push(box(width: cell, align(center + top,
        rotate(-90deg, reflow: true, text(size: 3.6pt, str(l))))))
    }
  }
  let cols = if labels { (auto,) + (cell,) * 16 } else { (cell,) * 16 }
  grid(columns: cols, row-gutter: 0pt, column-gutter: 0pt, ..cells)
}
#let _legend(vmax, neg, pos, w: 2.4cm) = stack(dir: ltr, spacing: 4pt,
  text(size: 6pt)[$-#calc.round(vmax, digits: 3)$],
  box(width: w, height: 5pt, radius: 1pt, fill: gradient.linear(
    (_mkrgb(neg), 0%), (white, 50%), (_mkrgb(pos), 100%))),
  text(size: 6pt)[$+#calc.round(vmax, digits: 3)$],
)
#let _hmpanel(g, neg, pos, title, cell: 7pt, layeraxis: false) = align(center, stack(spacing: 4pt,
  text(size: 8pt, weight: "bold", title),
  _heatmap(g, neg, pos, cell: cell, layeraxis: layeraxis),
  _legend(g.vmax, neg, pos),
))
#let _tg = _parsegrid(csv("data/template_grid.csv").slice(1), 0, 2, 3)   // module,module_key,layer,signed
#let _pc1 = _parsegrid(csv("data/pc1_loading.csv").slice(1), 0, 1, 2)    // module,layer,loading
#let _exrows = csv("data/cluster_example_grids.csv").slice(1)            // id,role,acc,mean_cos,module,layer,signed
#let _exadapters = (
  ("e4676550", "0.94", true), ("a0cfd6fb", "0.95", true), ("88a7c2a4", "0.94", true),
  ("e5050532", "0.55", false), ("76aaf26f", "0.54", false), ("0ed6e74a", "0.56", false),
)
// cetz canvases for the scalar panels
#let _screecanvas = cetz.canvas({
  let vr = csv("data/pca_variance.csv").slice(1, 11).map(r => (float(r.at(0)), float(r.at(1)) * 100))
  plot.plot(size: (5.4, 3), x-label: "principal component", y-label: "% of map variance",
    x-tick-step: 2, y-tick-step: 25, y-min: 0, {
    plot.add(vr, mark: "o", mark-size: .13, style: (stroke: rgb("#2b8cbe")),
      mark-style: (fill: rgb("#2b8cbe"), stroke: none))
  })
})
#let _histcanvas = cetz.canvas({
  let h = csv("data/pairwise_cos_hist.csv").slice(1).map(r => (float(r.at(2)), float(r.at(3))))
  plot.plot(size: (5.4, 3), x-label: "pairwise map cosine", y-label: "# adapter pairs",
    x-tick-step: 0.25, y-tick-step: 100, y-min: 0, {
    plot.add-bar(h, bar-width: 0.033, style: (fill: rgb("#41ae76"), stroke: 0.3pt + white))
  })
})
#let _cohesioncanvas = cetz.canvas({
  let cc = (:)
  for r in csv("data/cluster_cohesion.csv").slice(1) { cc.insert(r.at(0), float(r.at(1))) }
  plot.plot(size: (5, 3), y-label: "mean pairwise cosine", x-label: none,
    x-min: 0.4, x-max: 3.6, y-min: 0, y-max: 1, y-tick-step: 0.25,
    x-tick-step: none, x-ticks: ((1, "within hi"), (2, "within lo"), (3, "lo vs hi")), {
    plot.add-bar(((1, cc.at("within_high_acc")),), bar-width: 0.7, style: (fill: rgb("#41ae76"), stroke: none))
    plot.add-bar(((2, cc.at("within_low_acc")),), bar-width: 0.7, style: (fill: rgb("#d7301f"), stroke: none))
    plot.add-bar(((3, cc.at("low_vs_high")),), bar-width: 0.7, style: (fill: rgb("#999999"), stroke: none))
  })
})
