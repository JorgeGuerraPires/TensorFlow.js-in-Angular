import { Component } from '@angular/core';
import { BostonHousingDataset } from "./utils/data"

// // Boston Housing data constants:
// const BASE_URL =
//   'https://storage.googleapis.com/tfjs-examples/multivariate-linear-regression/data/';

import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';
import * as normalization from './utils/normalization';
import { Tensor2D, Tensor1D } from "@tensorflow/tfjs-core";




@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent {

  dataset: BostonHousingDataset = new BostonHousingDataset();


  //--------------------------------------------
  bostonData = new BostonHousingDataset();
  tensors = {};
  //--------------------------------------------

  baselineMsg = ""
  loadingData = true;

  title = 'sandbox';

  //---------------------
  x = [];
  y = []
  //-----------------------

  public graph = {
    data: [
      { x: this.x, y: this.y, type: 'scatter', mode: 'lines+points', marker: { color: 'red' } }],
    layout: { width: 320, height: 240, title: 'A Fancy Plot' }
  };

  constructor() {
    this.bostonData.loadData().then(() => {
      this.arraysToTensors();
      this.loadingData = false;
    });
  }
  // Convert loaded data into tensors and creates normalized versions of the
  // features.
  arraysToTensors() {

    const rawTrainFeatures = tf.tensor2d(this.bostonData.trainFeatures);
    const trainTarget = tf.tensor2d(this.bostonData.trainTarget);
    const rawTestFeatures = tf.tensor2d(this.bostonData.testFeatures);
    const testTarget = tf.tensor2d(this.bostonData.testTarget);
    const tensors = { rawTestFeatures, trainTarget, rawTrainFeatures, testTarget }

    // Normalize mean and standard deviation of data.
    let { dataMean, dataStd } =
      normalization.determineMeanAndStddev(tensors.rawTrainFeatures);

    const trainFeatures = normalization.normalizeTensor(
      tensors.rawTrainFeatures, dataMean, dataStd);
    const testFeatures =
      normalization.normalizeTensor(tensors.rawTestFeatures, dataMean, dataStd);

    this.tensors = { ...tensors, trainFeatures, testFeatures };

    this.computeBaseline(trainTarget, testTarget);


  };

  computeBaseline(trainTarget: Tensor2D, testTarget: Tensor2D) {
    const avgPrice = trainTarget.mean();
    console.log(`Average price: ${avgPrice.dataSync()}`);
    const baseline = testTarget.sub(avgPrice).square().mean();
    console.log(`Baseline loss: ${baseline.dataSync()}`);
    this.baselineMsg = `Baseline loss (meanSquaredError) is ${baseline.dataSync()[0].toFixed(2)}`;

  };


}