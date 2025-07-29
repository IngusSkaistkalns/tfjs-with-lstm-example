const tf = await import("@tensorflow/tfjs-node");

export const windowSize = 28;
export const lstmUnits = 96;
export const denseUnits = 64;
export const epochs = 10;
export const refit = 2;

export async function createModel(windowSize, lstmUnits) {
  const activation = "relu";
  const model = await tf.sequential();

  await model.add(
    tf.layers.lstm({
      units: lstmUnits,
      inputShape: [windowSize, 1],
      returnSequences: false,
    })
  );

  await model.add(tf.layers.dense({ units: 1, activation }));

  await model.compile({
    optimizer: tf.train.adam(0.01),
    loss: "meanSquaredError",
  });

  return model;
}

export async function fitModel(model, xs, ys, epochs = 10) {
  const history = await model.fit(xs, ys, {
    epochs,
    validationSplit: 0.15,
    callbacks: tf.callbacks.earlyStopping({
      patience: 6,
      restoreBestWeight: true,
    }),
  });

  return { model, history };
}

export function createTensors(data, windowSize) {
  const xs = [];
  const ys = [];

  for (let i = 0; i < data.length - windowSize; i++) {
    const x = data.slice(i, i + windowSize);
    const y = data.slice(i + windowSize, i + windowSize + 1);
    xs.push(x);
    ys.push(y.map((x) => x[0]));
  }

  return {
    xs: tf.tensor3d(xs, [xs.length, windowSize, 1]), // [samples, timeSteps, 1]
    ys: tf.tensor2d(ys), // [samples, forecastHorizon]
  };
}

export function normalize(v, vMin, vMax) {
  return (v - vMin) / (vMax - vMin);
}

export function denormalize(v, vMin, vMax) {
  return v * (vMax - vMin) + vMin;
}

export async function validateModel(model, xTrainValid) {
  const predictions = [];

  for (const slice of xTrainValid) {
    const inputTensor = tf.tensor3d([slice], [1, slice.length, 1]);
    const predictionTensor = model.predict(inputTensor);
    const nextValue = await predictionTensor.data();
    predictions.push(nextValue[0]);
  }

  return predictions;
}

export function buildValidationXTrain(matrix, normalizedFullData, windowSize) {
  var xTrainValid = [];

  for (
    let i = matrix.length - windowSize;
    i + windowSize < normalizedFullData.length;
    i++
  ) {
    xTrainValid.push(
      normalizedFullData.slice(i, i + windowSize).map((x) => [x])
    );
  }

  return xTrainValid;
}

export async function forecastFuture(
  rModel,
  matrix,
  windowSize,
  iterations,
  refit,
  refitWindow = 0
) {
  let matrixWithPredictions = matrix.slice(0);
  let currentWindow = matrix.slice(-windowSize);

  const predictions = [];

  for (let i = 0; i < iterations; i = i + 1) {
    const inputTensor = tf.tensor3d([currentWindow], [1, windowSize, 1]);
    const predictionTensor = rModel.predict(inputTensor);
    const nextValue = await predictionTensor.data();

    const prediction = [...nextValue][0];

    predictions.push(prediction);
    currentWindow.push([prediction]);
    matrixWithPredictions.push([prediction]);

    if (refit) {
      const series = createTensors(
        matrixWithPredictions.slice(-refitWindow),
        windowSize
      );
      await rModel.fit(series.xs, series.ys, {
        epochs: refit,
        shuffle: false,
      });
    }

    currentWindow.shift();
    console.log(`Iteration ${i + 1} of ${iterations}`);
  }

  return predictions;
}

console.log({ windowSize, lstmUnits, epochs, refit });

global.tf = tf;
global.createModel = createModel;
global.fitModel = fitModel;
global.normalize = normalize;
global.denormalize = denormalize;
global.createTensors = createTensors;
global.buildValidationXTrain = buildValidationXTrain;
global.validateModel = validateModel;
global.forecastFuture = forecastFuture;
global.windowSize = windowSize;
global.lstmUnits = lstmUnits;
global.denseUnits = denseUnits;
global.epochs = epochs;
global.refit = refit;
