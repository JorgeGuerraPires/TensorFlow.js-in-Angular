import { Component } from '@angular/core';
import { BostonHousingDataset } from "./utils/data"

// // Boston Housing data constants:
// const BASE_URL =
//   'https://storage.googleapis.com/tfjs-examples/multivariate-linear-regression/data/';

import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';
import * as normalization from './utils/normalization';
import { Tensor2D, Tensor1D } from "@tensorflow/tfjs-core";

import { NeuralnetworkService } from "./models/neuralnetwork.service";
import { element } from 'protractor';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent {

  // //--------------------------------------
  public graph = {
    data: [
      { x: [], y: [], type: 'scatter', mode: 'lines+points', marker: { color: 'red' } }
    ],
    layout: { width: 720, height: 740, title: 'A Fancy Plot' }
  };
  // //-------------------------------------

  dataset: BostonHousingDataset = new BostonHousingDataset();


  //--------------------------------------------
  bostonData = new BostonHousingDataset();
  tensors = {};
  //--------------------------------------------

  baselineMsg = ""
  loadingData = true;
  simulationRunning = false;

  title = 'sandbox';

  //---------------------
  x = [];
  y = [];
  //-----------------------

  // public graph = {
  //   data: [
  //     { x: this.x, y: this.y, type: 'scatter', mode: 'points', marker: { color: 'red' } }],
  //   layout: { width: 320, height: 240, title: 'A Fancy Plot' }
  // };

  constructor(private neuralnetwork: NeuralnetworkService) {


  }


  linearRegression() {
    /**may need to improve on this method */
    this.simulationRunning = true;
    this.bostonData.loadData().then(() => {
      this.arraysToTensors();
      this.loadingData = false;

    });
  }
  // Convert loaded data into tensors and creates normalized versions of the
  // features.
  private arraysToTensors() {


    // console.log(this.bostonData.trainFeatures);

    // const container2 = (document.querySelector(`#rawData .chart`) as HTMLInputElement);
    // tfvis.show.history(container2, this.bostonData.trainFeatures, ['loss'])


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

    const model = (this.neuralnetwork.linearRegressionModel(this.bostonData) as tf.Sequential);
    const modelName = "linear";

    const container = (document.querySelector(`#${modelName} .chart`) as HTMLInputElement);
    const containerstatus = (document.querySelector(`#${modelName} .status`) as HTMLInputElement);
    const containerStarted = (document.querySelector(`#${modelName} .started`) as HTMLInputElement);
    const inspectionHeadlineElement = (document.getElementById('inspectionHeadline') as HTMLInputElement);
    const table = (document.getElementById('myTable') as HTMLTableElement);


    const containers = { container, containerstatus, containerStarted, inspectionHeadlineElement, table };

    this.neuralnetwork.run(model, trainFeatures,
      trainTarget, testFeatures, testTarget, containers).then((aux) => {
        this.simulationRunning = false;

        const aux2 = this.bostonData.trainFeatures.map((element: any) => element[0]);
        const aux3 = this.bostonData.trainTarget.map((element: any) => element[0]);
        // const aux6 = aux.map((ele, i) => { return { x: i } })

        // console.log(aux[0]);
        // console.log(this.bostonData.trainTarget[0][0]);


        // this.graph.data = [
        //   { x: aux2, y: aux3, type: 'scatter', mode: 'markers', marker: { color: 'red' } }
        // ];

        const aux7 = aux2.map((element: any, index: number) => { return { x: element, y: aux3[index] } })

        //--------------------------------------------


        const aux8 = aux2.map((element: any, index: number) => { return { x: element, y: aux[index] } })

        let values = [aux7, aux8];

        const series = ['real data', 'Predicted by the NN'];
        tfvis.render.scatterplot(document.getElementById('plot4') as HTMLInputElement, { values, series }, {
          width: 700,
          height: 500,
          xLabel: 'Crime rate',
          yLabel: 'Median value of owner-occupied homes in units of $1,000'
        });
        //--------------------------------------------


        const aux9 = this.bostonData.trainFeatures.map((element: any) => element[1]);
        const aux10 = aux9.map((element: any, index: number) => { return { x: element, y: aux3[index] } });
        values = [aux10, aux8];

        // const series = ['real data', 'Predicted by the NN'];
        tfvis.render.scatterplot(document.getElementById('plot2') as HTMLInputElement, { values, series }, {
          width: 700,
          height: 500,
          xLabel: 'Proportion of residential land zoned for lots over 25,000 sq. ft.',
          yLabel: 'Median value of owner-occupied homes in units of $1,000'
        });





      });
    // const containers = { container, containerstatus, containerStarted };


    // this.neuralnetwork.predict(trainFeatures);

  };

  computeBaseline(trainTarget: Tensor2D, testTarget: Tensor2D) {
    const avgPrice = trainTarget.mean();
    console.log(`Average price: ${avgPrice.dataSync()}`);
    const baseline = testTarget.sub(avgPrice).square().mean();
    console.log(`Baseline loss: ${baseline.dataSync()}`);
    this.baselineMsg = `Baseline loss (meanSquaredError) is ${baseline.dataSync()[0].toFixed(2)}`;

  };

}