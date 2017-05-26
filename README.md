# NeuralNet_processing

Based on a <a href="https://github.com/fasc8/NeuralNet_C-">C++ tutorial</a>

<a href="/Rendered.md">Rendered Window / Explanation</a>

## Files
- <a href="/main/Connection.pde">Connection.pde</a> -> Class for handling the connections between the neurons  
- <a href="/main/Layer.pde">Layer.pde</a> -> Class with neuron array in it  
- <a href="/main/Net.pde">Net.pde</a> -> Class works as neural net  
- <a href="/main/Neuron.pde">Neuron.pde</a> -> Class for handling a neuron  
- <a href="/main/getData.pde">getData.pde</a> -> Contains the handling with inputdata and outputdata  
- <a href="/main/main.pde">main.pde</a> -> Handles all the other files and contains setup() and draw()

## Usage

Topology is the basic array for building the neural net.
Every item is a layer and its value is the number of neurons in it.

```processing
  //Create topology -> Item = Neurons in this layer
  int[] topology = new int[] { 4, 5, 6, 5};
  
  //So basicly
  int totalLayerNumber = topology.length; //4 layers
  int neuronNumber = topology[2]; //6 neurons
```

The next step is to feed the network with trainingdata

```processing
void draw() {
  background(51);
    inputVals = getNextInput(); //Get the next inputValues
    targetVals = calcOutput(inputVals); //Targetvalues get calculated here for this example
    //could be already set preprocessing like
    //targetVals = getNextTargets();
    
    myNet.feedForward(inputVals); //Feed the neural net
    resultVals = myNet.getResults(resultVals); //get the values calculated by the neural net
    myNet.backProp(targetVals); //Check the result against the target and readjust the weights
  }
  ```
There are a few functions we need to take a look at:  
  - <a href="https://github.com/fasc8/NeuralNet_processing#getnextinput">getNextInput / calcOutput</a>  
  - <a href="https://github.com/fasc8/NeuralNet_processing#feedforward">feedForward</a>  
  - <a href="https://github.com/fasc8/NeuralNet_processing#getresults">getResults</a>  
  - <a href="https://github.com/fasc8/NeuralNet_processing#backprop">backProp</a>  
    
### getNextInput

Here we calculate our input data. In this case it´s a random array with four items. Each item can be wether a one or a zero.  
The function returns an double array. The item count should fit to the neuron count in the first layer, the input layer.
```processing
//get random inputData
double[] getNextInput() {
  double[] inputVals = new double[] { round(random(0, 1)), round(random(0, 1)), round(random(0, 1)), round(random(0, 1)) };
  return inputVals;
}
//In this example we have a input layer with 4 neurons + 1 bias neuron
//That´s why inputVals has 4 items
```

And here we calculate the target output. We want to know how many zeros in the array are. There are five possible outputs. For example if there are two zeros in the input the output array looks like { 0, 0, 1, 0, 0 }  
The 1 is at the location where the corresponding number is. So the 1 in the example above means that there are two zeros(rendered as black squares/rectangles)
```processing
//Calculate the output
double[] calcOutput(double[] inputData) {
  int count = 0;
  double[] data = new double[] {};
  
  //Count the dark rectangles
  for(int i = 0; i < inputData.length; i++) {
    if(inputData[i] == 0) {
      count++;
    }
  }
  
  //set the targetValues
  for(int c = 0; c < inputData.length + 1; c++) {
    if(c == count) {
      data = (double[])append(data, 1);
    } else {
      data = (double[])append(data, 0);
    }
  }
  
  return data;
}
```
### feedForward
This function triggers the feedForward function for each neuron
```processing
  void feedForward(double[] inputVals) {
    assert(inputVals.length == m_layers[0].neuron.length - 1); //Assert that the inputValues are as many as neurons in the first layer
    
    for (int i = 0; i < inputVals.length; ++i) { //assign (latch) the input values into input neurons
      m_layers[0].neuron[i].setOutputVal(inputVals[i]);
    }
    
    for (int layerNum = 1; layerNum < m_layers.length; ++layerNum) { //Forward propagate
      Layer prevLayer = m_layers[layerNum - 1];
      for (int n = 0; n < m_layers[layerNum].neuron.length - 1; ++n) {
        m_layers[layerNum].neuron[n].feedForward(prevLayer);
      }
    }
  }
```
So here is the feedForward in a single neuron
```processing
  void feedForward(Layer prevLayer) {
    double sum = 0.0;
    //Sum the previous layer´s outputs (which are our inputs)
    //Include the bias node from the previous layer.

    for (int n = 0; n < prevLayer.neuron.length; ++n) {
      sum += prevLayer.neuron[n].getOutputVal() 
        * prevLayer.neuron[n].m_outputWeights[m_myIndex].weight;
    }

    m_outputVal = transferFunction(sum); //This is a function to limit the value between -1 and 1 and sets the outputvalue to it
  }
```
### getResults
Here we get the resultvalues that are calculated by the neural network
```processing
  double[] getResults(double[] resultVals) {
    resultVals = new double[] {};

    for (int n = 0; n < m_layers[m_layers.length - 1].neuron.length - 1; ++n) {
      resultVals = (double[])append(resultVals, m_layers[m_layers.length - 1].neuron[n].getOutputVal());
      //Here we get the outputvalues of the neurons in the last layer, the output layer
    }
    return resultVals;
  }
```
### backProp
Here we readjust the weights of the single connections to get better results
```processing
  void backProp(double[] targetVals) {
    //Calculate overall net error (RMS of output neuron errors)
    Layer outputLayer = m_layers[m_layers.length - 1];
    m_error = 0.0;

    for (int n = 0; n < outputLayer.neuron.length - 1; ++n) {
      double delta = targetVals[n] - outputLayer.neuron[n].getOutputVal();
      m_error += delta * delta;
    }
    m_error /= outputLayer.neuron.length - 1; //get average error squared
    m_error = Math.sqrt(m_error); //rms

    //Implement a recent average measurment:
    m_recentAverageError = (m_recentAverageError * m_recentAverageSmoothingFactor + m_error) / (m_recentAverageSmoothingFactor + 1.0);

    //Claculate output layer gradients
    for (int n = 0; n < outputLayer.neuron.length - 1; ++n) {
      outputLayer.neuron[n].calcOutputGradients(targetVals[n]);
    }

    //Calculate gradients on hidden layers
    for (int layerNum = m_layers.length - 2; layerNum > 0; --layerNum) {
      Layer hiddenLayer = m_layers[layerNum];
      Layer nextLayer = m_layers[layerNum + 1];

      for (int n = 0; n < hiddenLayer.neuron.length; ++n) {
        hiddenLayer.neuron[n].calcHiddenGradients(nextLayer);
      }
    }

    //For all layers from outputs to first hidden layer
    //update connection weights
    for (int layerNum = m_layers.length - 1; layerNum > 0; --layerNum) {
      Layer layer = m_layers[layerNum];
      Layer prevLayer = m_layers[layerNum - 1];

      for (int n = 0; n < layer.neuron.length - 1; ++n) {
        layer.neuron[n].updateInputWights(prevLayer);
      }
    }
  }
```
