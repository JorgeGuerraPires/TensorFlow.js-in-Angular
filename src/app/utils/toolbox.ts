import * as tf from '@tensorflow/tfjs';
import { featureDescriptions } from "../utils/data"

/**
 * Describe the current linear weights for a human to read.
 *
 * @param {Array} kernel Array of floats of length 12.  One value per feature.
 * @returns {List} List of objects, each with a string feature name, and value
 *     feature weight.
 */
export function describeKernelElements(kernel: string | any[] | Float32Array | Int32Array | Uint8Array) {
    // tf.util.assert(
    //     kernel.length == 12,
    //     `kernel must be a array of length 12, got ${kernel.length}`);

    const outList = [];
    for (let idx = 0; idx < kernel.length; idx++) {
        outList.push({ description: featureDescriptions[idx], value: kernel[idx] });
    }
    return outList;
}

const NUM_TOP_WEIGHTS_TO_DISPLAY = 5;

/**
 * Updates the weights output area to include information about the weights
 * learned in a simple linear model.
 * @param {List} weightsList list of objects with 'value':number and
 *     'description':string
 */
export function updateWeightDescription(weightsList: any[], inspectionHeadlineElement: HTMLElement, table: HTMLTableElement) {

    inspectionHeadlineElement.innerText =
        `Top ${NUM_TOP_WEIGHTS_TO_DISPLAY} weights by magnitude`;

    // Sort weights objects by descending absolute value.
    weightsList.sort((a, b) => Math.abs(b.value) - Math.abs(a.value));
    // var table = document.getElementById('myTable');
    // Clear out table contents
    table.innerHTML = '';
    // Add new rows to table.
    weightsList.forEach((weight, i) => {
        if (i < NUM_TOP_WEIGHTS_TO_DISPLAY) {
            let row = table.insertRow(-1);
            let cell1 = row.insertCell(0);
            let cell2 = row.insertCell(1);
            if (weight.value < 0) {
                cell2.setAttribute('class', 'negativeWeight');
            } else {
                cell2.setAttribute('class', 'positiveWeight');
            }
            cell1.innerHTML = weight.description;
            cell2.innerHTML = weight.value.toFixed(4);
        }
    });

};