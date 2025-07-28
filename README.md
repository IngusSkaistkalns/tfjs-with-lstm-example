# TFJS with LSTM layer example

For Mitigate knowledge sharing

### Run scripts

Run scripts and start console:

```bash
$ npm run dev
```

**See globals in functions and data**

### TRAIN MODEL AND COPY WEIGHTS

```js
await fitModel(model, xs, ys, epochs);
cModel.setWeights(model.getWeights());
```

### VALIDATE MODEL MANUALLY

```js
predictions = await validateModel(cModel, xTrainValid);
pdiff = data[dataK.length - 1] - denormalize(predictions[0], vMin, vMax);
predictions.forEach((x) => console.log(x * (vMax - vMin) + vMin + pdiff));
```

### FORECAST FUTURE

```js
forecasts = await forecastFuture(cModel, matrix, windowSize, 60, 2, 90);
fdiff = data[dataK.length - 1] - denormalize(forecasts[0], vMin, vMax);
forecasts.forEach((x) => console.log(x * (vMax - vMin) + vMin + fdiff));
cModel.setWeights(model.getWeights());
null;
```
