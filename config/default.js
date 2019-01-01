module.exports = {
  compile : {
    optimizer: 'rmsprop',
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
  },
  fit: {
    batchSize: 200,
    validationSplit: 0.10,
    epochs: 8
  }
}
