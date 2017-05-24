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
    - <a href="/images/MainFrame.png">getNextInput</a>
    - <a href="/images/MainFrame.png">feedForward</a>
    - <a href="/images/MainFrame.png">getResults</a>
    - <a href="/images/MainFrame.png">backProp</a>
