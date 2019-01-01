const fs = require('fs');
var jpeg = require('jpeg-js');
const tf = require('@tensorflow/tfjs');
require('@tensorflow/tfjs-node');

// define dimensions of data

const width = 32;
const height = 32;
const layers = 3;
const imgBufferLen = width*height*layers;

module.exports = {
  loadBatch: loadBatch,
  drawBatch: drawBatch,
  loadJPEG: loadJPEG,
  labelMap: fs.readFileSync(__dirname + '/cifar-10-batches-bin/batches.meta.txt').toString().split("\n"),
  drawArray: drawArray
}


function loadBatch(batchN, imgN = Array.from(Array(10000).keys())) {

  console.log('Reading data from file..');
  
  // Binary data downloaded from
  // http://www.cs.toronto.edu/~kriz/cifar.html
  
  var batchFile = __dirname + '/cifar-10-batches-bin/data_batch_'+ batchN +'.bin';

  var batchData = fs.readFileSync(batchFile);


  var labels = [];
  var images = [];

  imgN.forEach (function(i) {
    var labelByte = i*(1+imgBufferLen);
    var imgStartByte = labelByte + 1;
    
    var labelArr = Array(10).fill(0);
    labelArr[batchData[labelByte]] = 1;
    labels.push(labelArr);
    
    images.push(imgBuffer2Array(batchData.slice(imgStartByte,imgStartByte + imgBufferLen),width, height));
  });

  console.log('Creating input tensors..');

  module.exports.y = tf.tensor2d(labels);
  module.exports.x = tf.tensor4d(images);
  
  //~ module.exports.y = tf.tensor2d(labels, [10000,10]);
  //~ module.exports.x = tf.tensor4d(images, [10000,width,height,layers]);
}

function imgBuffer2Array(buffer, width, height) {
  var rStart = 0;
  var gStart = width*height;
  var bStart = 2*width*height;
  
  var image = [];
  for (row = 0; row < height; row++) {
    var rowpixels = [];
    for (col = 0; col < width; col++) {
      var values = [
        buffer[rStart + row*width + col],
        buffer[gStart + row*width + col],
        buffer[bStart + row*width + col]
      ]
      rowpixels.push(values);
    }
    image.push(rowpixels);
  }
  return image;
}

function loadJPEG(jpegData) {
  var {data, width, height} = jpeg.decode(jpegData);
  var image = [];
  for (row = 0; row < height; row++) {
    var rowpixels = [];
    for (col = 0; col < width; col++) {
      var values = [
        data[row*width*(layers+1) + col*(layers+1) + 0],
        data[row*width*(layers+1) + col*(layers+1) + 1],
        data[row*width*(layers+1) + col*(layers+1) + 2]
      ];
      rowpixels.push(values);
    }
    image.push(rowpixels);
  }
  // transform to tensor
  module.exports.x = tf.tensor4d([image]);
}


function drawBatch(batchN, imgNum = Array.from(Array(10000).keys())) {
  var batchFile = __dirname + '/cifar-10-batches-bin/data_batch_'+ batchN +'.bin';
  var batchData = fs.readFileSync(batchFile);
  
  imgNum.forEach (function(imgN) {
    var fileout = './test-jpeg/batch_'+batchN+'_img_'+imgN+'.jpg'
    
    var imgStartByte = imgN*(1+imgBufferLen) + 1;
    
    var rStart = imgStartByte + 0;
    var gStart = imgStartByte + width*height;
    var bStart = imgStartByte + 2*width*height;
    
    var frameData = [];
    
    for (i=0; i < width*height; i++) {
      frameData.push(batchData[rStart+i]); // red
      frameData.push(batchData[gStart+i]); // green
      frameData.push(batchData[bStart+i]); // blue
      frameData.push(0xFF); // alpha - ignored in JPEGs
    }
    
    var rawImageData = {
      data: frameData,
      width: width,
      height: height
    };
    var jpegImageData = jpeg.encode(rawImageData, 100);
    
    fs.writeFileSync(fileout, jpegImageData.data);
  });
}

function drawArray(imageArr, fileout) {
  //draw a jpeg from the internal array representation (mainly for verification purposes)
  
  var frameData = [];
  
  imageArr.forEach(function(row) {
    row.forEach(function(pix) {
      frameData.push(pix[0]);
      frameData.push(pix[1]);
      frameData.push(pix[2]);
      frameData.push(0xFF); // alpha ignored
    });
  });
  
  var rawImageData = {
    data: frameData,
    width: width,
    height: height
  };
  var jpegImageData = jpeg.encode(rawImageData, 100);

  fs.writeFileSync(fileout, jpegImageData.data);
}
