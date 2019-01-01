//global modules
var resolvePath = require('path').resolve;

//installed modules
const tf = require('@tensorflow/tfjs');
require('@tensorflow/tfjs-node');

// local modules
//~ // data
var data = require('./data/cifar');

// PARAMETERS
const modelDir = __dirname + '/models/saved-models/defaultmodel';


module.exports = predict;


function predict(jpegData) {
  return new Promise(function (resolve, reject) {
    //img to tf.tensor
    data.loadJPEG(jpegData);
    //predict
    tf.loadModel('file://'+resolvePath(modelDir+'/model.json')).then(function(model) {
      var p = model.predict(data.x).dataSync();
      //zip a table with labels
      var predictions = [];
      for (var i = 0 ; i < p.length ; i++) {
        predictions.push([p[i], data.labelMap[i]]);
      }
      //sort
      predictions.sort(function(a,b) {
        return (a[0]<b[0]);
      });
      
      //resolve promise
      resolve(predictions);
    });
  });
}





