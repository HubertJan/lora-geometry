// Native-Typst schematic of the LoL-based meta-model architecture (drawn with fletcher).
// One LoRA matrix pair (B, A) of a single component is processed top-to-bottom:
// equivariant reduction -> product B'A' -> flatten -> concat with the other components'
// vectors -> regression head. Included by sections/approach.typ.
#import "@preview/fletcher:0.5.8" as fletcher: diagram, node, edge

// A plain matrix "chip": a stroked rectangle of the given absolute size with an
// optional italic label centred inside it.
#let _mat(pos, w, h, name: none, fill: none, label: none) = node(
  pos,
  if label != none { text(9pt, label) } else { [] },
  shape: rect,
  width: w,
  height: h,
  inset: 0pt,
  fill: fill,
  stroke: 0.7pt,
  name: name,
)

#let lol-diagram = align(center, diagram(
  spacing: (7mm, 9mm),
  node-stroke: none,
  {
    // ---- Stage 1: raw LoRA pair for one component -------------------------
    node((0, -0.7), text(8.5pt, weight: "bold")[Layer 1 · Attn K])
    _mat((-0.55, 0), 8mm, 20mm, name: <B>, fill: blue.lighten(80%), label: $B$)
    node((-0.25, 0), text(7pt)[$ in RR^(m times r)$])
    _mat((0.55, 0), 20mm, 8mm, name: <A>, fill: red.lighten(80%), label: $A in RR^(r times n)$)
    node((-0.27, 0.74), text(7pt)[$W_A B = B'$])
    node((0.85, 0.74), text(7pt)[$W_A A = A'$])

    // ---- Stage 2: equivariant linear layer (independent reduction) --------
    _mat((-0.55, 1.5), 8mm, 11mm, name: <Bp>, fill: blue.lighten(80%), label: $B'$)
    node((-0.25, 1.5), text(7pt)[$ in RR^(m times r)$])
    _mat((0.55, 1.5), 11mm, 8mm, name: <Ap>, fill: red.lighten(80%), label: $A' $)
    node((0.90, 1.5), text(7pt)[$ in RR^(m times r)$])
    edge(<B>, <Bp>, "->")
    edge(<A>, <Ap>, "->")
    node((0, 1.1), text(6.5pt, fill: gray.darken(30%))[equivariant linear layer])

    // ---- Stage 3: form the product B'A' -----------------------------------
    _mat((-0.28, 3.0), 8mm, 11mm, name: <Bm>, fill: blue.lighten(80%), label: $B'$)
    node((0, 3.0), text(10pt)[$times$])
    _mat((0.28, 3.0), 11mm, 8mm, name: <Am>, fill: red.lighten(80%), label: $A'$)
    edge(<Bp>, <Bm>, "->")
    edge(<Ap>, <Am>, "->")

    // ---- Stage 4: the product square --------------------------------------
    _mat((0, 4.1), 11mm, 11mm, name: <P>, fill: purple.lighten(82%), label: $B'A'$)
    node((0.35, 4.1), text(7pt)[$in RR^(d times d)$])
    edge((0, 3.0), <P>, "->")

    // ---- Stage 5: flatten + L2-normalise to a per-pair vector -------------
    _mat((0, 5.3), 4mm, 16mm, name: <x1>, fill: purple.lighten(82%))
    edge(<P>, <x1>, "->", label: text(6.5pt)[flatten + $ell_2$], label-side: right)
    node((0.2, 5.3), text(7pt)[$ in RR^(d^2)$])

    // ---- Stage 6: the other components' vectors, concatenated -------------
    _mat((-0.9, 6.9), 4mm, 16mm, name: <x0>, fill: purple.lighten(88%))
    node((-0.9, 7.7), text(6.5pt)[L1 Attn Q])
    _mat((0, 6.9), 4mm, 16mm, name: <x1b>, fill: purple.lighten(82%))
    node((0, 7.7), text(6.5pt)[L1 Attn K])
    _mat((0.9, 6.9), 4mm, 16mm, name: <x2>, fill: purple.lighten(88%))
    node((0.9, 7.7), text(6.5pt)[L1 MLP-Up])
    node((1.55, 6.9), text(9pt)[$dots.c$])
    edge(<x1>, <x1b>, "->")

    // brace-like grouping into one concatenated feature vector
    _mat((0, 8.7), 34mm, 6mm, name: <cat>, fill: purple.lighten(78%),
      label: text(8pt)[$x = [x_1, x_2, dots, x_c]$])
    edge(<x0>, <cat>, "->")
    edge(<x1b>, <cat>, "->")
    edge(<x2>, <cat>, "->")

    // ---- Stage 7: regression head ----------------------------------------
    node((0, 9.7), text(8.5pt, weight: "bold")[Regression head],
      shape: fletcher.shapes.pill, fill: green.lighten(80%), stroke: 0.7pt, name: <head>)
    edge(<cat>, <head>, "->")
    node((0, 10.5), text(8pt)[decision / value $hat(y)$], name: <out>)
    edge(<head>, <out>, "->")
  },
))
