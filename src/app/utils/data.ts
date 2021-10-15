
import { Component } from '@angular/core';
import { Papa } from 'ngx-papaparse';

// Boston Housing data constants:
const BASE_URL =
    'https://storage.googleapis.com/tfjs-examples/multivariate-linear-regression/data/';

export const featureDescriptions = [
    'Crime rate', 'Land zone size', 'Industrial proportion', 'Next to river',
    'Nitric oxide concentration', 'Number of rooms per house', 'Age of housing',
    'Distance to commute', 'Distance to highway', 'Tax rate', 'School class size',
    'School drop-out rate'
];


/** Helper class to handle loading training and test data. */
export class BostonHousingDataset {

    TRAIN_FEATURES_FN = 'train-data.csv';
    TRAIN_TARGET_FN = 'train-target.csv';
    TEST_FEATURES_FN = 'test-data.csv';
    TEST_TARGET_FN = 'test-target.csv';

    trainFeatures: any;
    trainTarget: any;
    testFeatures: any;
    testTarget: any;
    papa: Papa = new Papa()

    constructor() { }

    getData() {
        if (!this.trainFeatures)
            this.loadData().then(() => console.log(this.trainFeatures));
        console.log(this.trainFeatures);

        return this.trainFeatures;
    }

    /** Loads training and test data. */
    async loadData() {

        [this.trainFeatures, this.trainTarget, this.testFeatures, this.testTarget] =
            await Promise.all([
                this.loadCsv(this.TRAIN_FEATURES_FN), this.loadCsv(this.TRAIN_TARGET_FN),
                this.loadCsv(this.TEST_FEATURES_FN), this.loadCsv(this.TEST_TARGET_FN)
            ]);


        this.shuffle(this.trainFeatures, this.trainTarget);
        this.shuffle(this.testFeatures, this.testTarget);
    }

    /**
 * Downloads and returns the csv.
 *
 * @param {string} filename Name of file to be loaded.
 *
 * @returns {Promise.Array<number[]>} Resolves to parsed csv data.
 */
    loadCsv = async (filename: any) => {
        return new Promise(resolve => {
            const url = `${BASE_URL}${filename}`;

            console.log(`  * Downloading data from: ${url}`);
            this.papa.parse(url, {
                download: true,
                header: true,
                complete: (results: any) => {
                    // console.log(results);
                    resolve(this.parseCsv(results['data']));
                }
            })
        });
    };

    /**
* Given CSV data returns an array of arrays of numbers.
*
* @param {Array<Object>} data Downloaded data.
*
* @returns {Promise.Array<number[]>} Resolves to data with values parsed as floats.
*/
    parseCsv = async (data: any) => {
        return new Promise(resolve => {
            data = data.map((row: any) => {
                return Object.keys(row).map(key => parseFloat(row[key]));
            });
            resolve(data);
        });
    };

    /**
 * Shuffles data and target (maintaining alignment) using Fisher-Yates
 * algorithm.flab
 */
    shuffle(data: any, target: any) {
        let counter = data.length;
        let temp = 0;
        let index = 0;
        while (counter > 0) {
            index = (Math.random() * counter) | 0;
            counter--;
            // data:
            temp = data[counter];
            data[counter] = data[index];
            data[index] = temp;
            // target:
            temp = target[counter];
            target[counter] = target[index];
            target[index] = temp;
        }
    };

    get numFeatures() {
        // If numFetures is accessed before the data is loaded, raise an error.
        if (this.trainFeatures == null) {
            throw new Error('\'loadData()\' must be called before numFeatures')
        }
        return this.trainFeatures[0].length;
    }


}