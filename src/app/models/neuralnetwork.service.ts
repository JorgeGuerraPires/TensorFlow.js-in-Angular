/**
 * Here I am organizing several models for Neural Network using TensorFlow.js
 */
import { Injectable } from '@angular/core';
import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';
import { describeKernelElements, updateWeightDescription } from "../utils/toolbox"

import { Tensor2D, Tensor1D, Tensor } from "@tensorflow/tfjs-core";


import { BostonHousingDataset } from "../utils/data";

@Injectable({
  providedIn: 'root'
})
export class NeuralnetworkService {
  // Some hyperparameters for model training.
  NUM_EPOCHS = 200;
  BATCH_SIZE = 40;
  LEARNING_RATE = 0.01;

  constructor() { }

  /**
 * Builds and returns Linear Regression Model.
 *
 * @returns {tf.Sequential} The linear regression model.
 */
  linearRegressionModel(bostonData: BostonHousingDataset): tf.Sequential {
    const model = tf.sequential();
    model.add(tf.layers.dense({ inputShape: [bostonData.numFeatures], units: 1 }));

    model.summary();
    return model;
  };


  /**
 * Compiles `model` and trains it using the train data and runs model against
 * test data. Issues a callback to update the UI after each epcoh.
 *
 * @param {tf.Sequential} model Model to be trained.
 * @param {boolean} weightsIllustration Whether to print info about the learned
 *  weights.
 */
  async run(model: tf.Sequential, trainFeatures: Tensor,
    trainTarget: Tensor, testFeatures: Tensor, testTarget: Tensor,
    containers: { container: HTMLInputElement, containerstatus: HTMLInputElement, containerStarted: HTMLInputElement, inspectionHeadlineElement: HTMLInputElement, table: HTMLTableElement }) {

    const { container, containerstatus, containerStarted, inspectionHeadlineElement, table } = containers;

    model.compile(
      { optimizer: tf.train.sgd(this.LEARNING_RATE), loss: 'meanSquaredError' });

    let trainLogs: any[] = [];

    const tensors = { trainFeatures, trainTarget, testFeatures, testTarget };

    await model.fit(tensors.trainFeatures, tensors.trainTarget, {
      batchSize: this.BATCH_SIZE,
      epochs: this.NUM_EPOCHS,
      validationSplit: 0.2,//Split some of the training data off and use it as validation data
      //20% of the training data to use as validation data
      callbacks: {
        onTrainBegin: () => { containerStarted.innerText = "we just started training your network!😃😃😃" },
        onEpochEnd: async (epoch, logs) => {
          trainLogs.push(logs);
          tfvis.show.history(container, trainLogs, ['loss', 'val_loss']);

          const weightsIllustration = true;

          if (weightsIllustration) {
            model.layers[0].getWeights()[0].data().then(kernelAsArr => {
              const weightsList = describeKernelElements(kernelAsArr);
              updateWeightDescription(weightsList, inspectionHeadlineElement, table);
            });
          }


        },

      }
    });


    /**Note that the fit use train data, whereas the evaluation uses  test data */
    const result = (model.evaluate(
      tensors.testFeatures, tensors.testTarget, { batchSize: this.BATCH_SIZE }) as Tensor);

    const testLoss = result.dataSync()[0];

    const trainLoss = trainLogs[trainLogs.length - 1].loss;
    const valLoss = trainLogs[trainLogs.length - 1].val_loss;

    containerstatus.innerText =
      `Final train-set loss: ${trainLoss.toFixed(4)}\n` +
      `Final validation-set loss: ${valLoss.toFixed(4)}\n` +
      `Test-set loss: ${testLoss.toFixed(4)}`;

    return this.predict(model, trainFeatures, trainTarget);

    // return model;
  };//end of async run


  predict(model: tf.Sequential, dataset: Tensor, target: Tensor) {
    // console.log("some predictions");

    const prediction = (model.predict(dataset) as tf.Tensor);

    return prediction.dataSync();
    // prediction.print();
    // console.log(prediction.dataSync());
    // console.log(target.dataSync());


  }


}
