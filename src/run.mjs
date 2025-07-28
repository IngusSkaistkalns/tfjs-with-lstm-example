import "./functions.mjs";
import "./data.mjs";
import repl from "repl";

console.log(`
 ### TRAIN MODEL
 await fitModel(model, xs, ys, epochs);cModel.setWeights(model.getWeights());null

 ### VALIDATE MODEL MANUALLY
 predictions = await validateModel(cModel, xTrainValid)
 pdiff = data[dataK.length - 1] - denormalize(predictions[0], vMin, vMax)
 predictions.forEach((x) => console.log(x * (vMax - vMin) + vMin + pdiff));

 ### FORECAST FUTURE
 forecasts = await forecastFuture(cModel, matrix, windowSize, 60, 2, 90)
 fdiff = data[dataK.length - 1] - denormalize(forecasts[0], vMin, vMax)
 forecasts.forEach((x) => console.log(x * (vMax - vMin) + vMin + fdiff));
 cModel.setWeights(model.getWeights());null
`);

repl.start("> ");
