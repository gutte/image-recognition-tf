Basic image recognition (classification) using tensorflow.js.

- training script `training.js`
- model definitions `./models/`
- data controller and image processing `./data/`
- prediction function `predict.js`


The prediction function is exported in `index.js`. If the model has been
trained it can be used by other apps by linking the package locally
using `npm link`.
