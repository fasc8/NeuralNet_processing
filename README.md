# NeuralNet_processing

Based on a <a href="https://github.com/fasc8/NeuralNet_C-">C++ tutorial</a>

<a href="/images/MainFrame.png">Rendered</a>

# Usage

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
    targetVals = calcOutput(inputVals); //Targetvalues get calculated in this setup
    //could be already set preprocessing like
    targetVals = getNextTargets();
    
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
    
## getNextInput

Here we calculate our input data. In this case it´s a random array with four items. Each item can be wether a one or a zero.
```processing
//get random inputData
double[] getNextInput() {
  double[] inputVals = new double[] { round(random(0, 1)), round(random(0, 1)), round(random(0, 1)), round(random(0, 1)) };
  return inputVals;
}
```

And here we calculate the target output. We want to know how many zeros in the array are. There are five possible outputs. For example if there are two zeros in the input the output array looks like { 0, 0, 1, 0, 0 }
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
## feedForward

## getResults

## backProp

