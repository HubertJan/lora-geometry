// Seminar report — typeset with tracl (ACL Typst style).
// See https://typst.app/universe/package/tracl for details.
//
// The body is split into one file per section under sections/. Plot helpers live
// in their own modules (loto-scatter.typ, lrp-maps.typ) and are imported by the
// section files that use them.

#import "@preview/tracl:0.8.1": *
#import "@preview/pergamon:0.7.1": *


#show: doc => acl(doc,
  anonymous: false,
  title: [Trends in AI and Large Language Models
Summer Term 2026 - Seminar Report - Functional LoRA Geometry],
  authors: make-authors(
    (
      name: "Hubert Tomaszczak",
      affiliation: [Hasso-Plattner Institute\ #email("hubert.tomaszczak@student.hpi.de")],
    ),
  ),
)

#abstract[
  A natural assumption behind weight-space methods is that LoRA space is
  semantically organised: similar tasks should induce similar adapters, and
  geometric proximity should track functional similarity. We test this by
  training adapter pools that vary systematically in training dataset and
  conditions, fitting GL-equivariant meta models that predict SST2 performance
  from adapter weights alone, and evaluating them under leave-one-task-out
  shift. Within a pool the geometry looks informative: held-out adapters are
  predicted almost exactly ($R^2 = 0.84$, $rho = 0.95$), and a ridge on a few
  scalar weight-geometry features already recovers most of that calibration.
  Across unseen datasets only the ranking survives (median $rho = 0.68$) while
  calibration collapses ($R^2 <= 0$ on 6 of 15 datasets). Interpretability
  analyses show the capability is redundantly encoded across components and
  that the meta model looks only at a subset of components and probes the components using the decision directions. Geometry does not fail for lack of any structure. The structure is organised rather by the training setup
  and dataset, instead of capability.
]

#include "sections/introduction.typ"
#include "sections/related-work.typ"
#include "sections/approach.typ"
#include "sections/experimental_setup.typ"
#include "sections/results.typ"
#include "sections/limitations.typ"
#include "sections/conclusion.typ"

// Bibliography — add entries to custom.bib and cite with #cite(<key>).
#add-bib-resource(read("custom.bib"))
#print-acl-bibliography()

#appendix[
  #include "sections/appendix.typ"
]
