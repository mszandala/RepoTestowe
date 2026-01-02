#import "@preview/lilaq:0.5.0" as lq

#let meshsamples = 200
#let contoursamples = 200

#let raport = read("raport.log").split("\n")
#let fn-math = eval(mode: "math", raport.remove(0))
#let fn-plot = eval(mode: "code", raport.remove(0))
#let boundaries = csv(bytes(raport.remove(0))).at(0)

#set text(size: 12pt, lang: "pl", font: "New Computer Modern")

= Aquila Optimizer

Optymalizowana funkcja: #fn-math

#let fitness-line = raport.filter(x => x.starts-with("Fitness")).join("\n")
#let fitness-value = fitness-line.replace(regex("Fitness = (.+?), Position .*"), matches => {matches.captures.at(0)})
#let position-value = fitness-line.replace(regex("Fitness = (.+?), Position = \[(.*)\]"), matches => {matches.captures.at(1)})

Wynik końcowy: $f(#eval(mode: "math", position-value)) = #eval(mode: "math", fitness-value)$

#let csv-data = csv(bytes(raport.filter(x => not x.starts-with("Fitness")).join("\n")))
#let processed-data = (:)
#for value in csv-data {
  let x = processed-data.at(value.at(0), default: ())
  x.push((float(value.at(1)), float(value.at(2))))
  processed-data.insert(value.at(0), x)
}

#let colormesh = lq.colormesh(
  lq.linspace(float(boundaries.at(0)), float(boundaries.at(1)), num: meshsamples),
  lq.linspace(float(boundaries.at(2)), float(boundaries.at(3)), num: meshsamples),
  fn-plot,
  map: color.map.viridis,
)
#let contour = lq.contour(
  lq.linspace(float(boundaries.at(0)), float(boundaries.at(1)), num: contoursamples),
  lq.linspace(float(boundaries.at(2)), float(boundaries.at(3)), num: contoursamples),
  fn-plot,
  map: (rgb("#FFFFFF"), rgb("#FFFFFF")),
  levels: 7,
)

#for key in processed-data.keys() [
  = Iteracja #key
  #figure(
    lq.diagram(
      width: 7.5cm,
      height: 7.5cm,
      colormesh, contour,
      xlim: (float(boundaries.at(0)), float(boundaries.at(1))),
      ylim: (float(boundaries.at(2)), float(boundaries.at(3))),
      ..(
        processed-data
          .at(key)
          .map(value => {
            let (x, y) = value
            lq.ellipse(x, y, width: 0.25, height: 0.25, fill: red)
          })
      ),
    ),
    caption: [Iteracja #key],
  )
]
