const tf = require('@tensorflow/tfjs');
require('@tensorflow/tfjs-node');

module.exports = {
    create: create
}

function create(batchInputShape) {
  console.log('Creating model..');
  const model = tf.sequential();

  model.add(tf.layers.conv2d({
    kernelSize: 4,
    filters: 16,
    activation: 'relu',
    batchInputShape: batchInputShape
  }));

  model.add(tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }));

  model.add(tf.layers.conv2d({
    kernelSize: 4,
    filters: 32,
    activation: 'relu'
  }));

  model.add(tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }));

  model.add(tf.layers.conv2d({
    kernelSize: 4,
    filters: 32,
    activation: 'relu'
  }));

  model.add(tf.layers.flatten({}));

  model.add(tf.layers.dense({units: 64, activation: 'relu'}));

  model.add(tf.layers.dense({units: 10, activation: 'softmax'}));

  return model;
}
