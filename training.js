//global modules
var resolve = require('path').resolve;

//installed modules
const tf = require('@tensorflow/tfjs');
require('@tensorflow/tfjs-node');

// local modules
// data
var data = require('./data/cifar');
// uncompiled model files
var modelDef = require('./models/conv_v1');
// training config
const config = require('./config/default.js');

// PARAMETERS
const saveDir = './models/saved-models/defaultmodel';




// start

data.loadBatch(1);

var model = modelDef.create(data.x.shape);

console.log('Compiling model..');
model.compile(config.compile);

console.log('Fitting model..');
model.fit(data.x, data.y, config.fit).then(function (history) {
  console.log('Training completed.');
  console.log(history);
  console.log('Saving model');
  model.save('file://'+resolve(saveDir)).then(function (msg) {
    console.log(msg);
    console.log('Done');
  });
});
