//Net class
class Net {
  //Layer Array
  Layer[] m_layers = new Layer[] {}; // m_layers[layerNum][neuronNum]
  
  //Error Variables
  double m_error;
  double m_recentAverageError;
  double m_recentAverageSmoothingFactor;
  
  //Render Variables
  int scale = 200; //Neuron Scale
  int radius = 20; //Neuron Radius
  float sizeScale = 0.5; //scaleFactor
  float startPosX = 200; //Startposition x + scale * sizeScale
  float startPosY = 100; //Startposition y + 3 * radius
  
  //Net Constructor
  Net(int[] topology) {
    //number of Layers
    int numLayers = topology.length;
    
    //create layers
    for (int layerNum = 0; layerNum < numLayers; ++layerNum) {
      //add layer
      m_layers = (Layer[])append(m_layers, new Layer());
      //Calculate number of outputs + 1 bias
      int numOutputs = (layerNum == topology.length - 1) ? 0 : topology[layerNum + 1];
      
      //we have made a new Layer, now fill it with neurons, and
      //add a bias neuron to the layer
      for (int neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum) {
        m_layers[m_layers.length - 1].neuron = (Neuron[])append(m_layers[m_layers.length - 1].neuron, new Neuron(numOutputs, neuronNum));
       }

      //force the bias node´s output value to 1.0. Its the last neuron created above
      m_layers[m_layers.length - 1].neuron[m_layers[m_layers.length - 1].neuron.length - 1].setOutputVal(1.0);
    }
    
    //Set the render coordinates for the neurons
    for(int layerNum = 0; layerNum < m_layers.length; ++layerNum) {
      Layer layer = m_layers[layerNum];
      for (int neuronNum = 0; neuronNum < layer.neuron.length; ++neuronNum) {
        Neuron neuron = layer.neuron[neuronNum];
        neuron.x = startPosX + (layerNum + 1) * scale * sizeScale;
        neuron.y = startPosY + (neuron.m_myIndex + 1) * 3 * radius;
      }
    }
  }
  
  //Print all weights
  void printWeights() {
    for(int layerNum = 0; layerNum < m_layers.length; ++layerNum) {
      Layer layer = m_layers[layerNum];
      println("Layer: " + layerNum);
      for (int neuronNum = 0; neuronNum < layer.neuron.length; ++neuronNum) {
        Neuron neuron = layer.neuron[neuronNum];
        println("Neuron: " + neuronNum);
        for(int weiNum = 0; weiNum < neuron.m_outputWeights.length; weiNum++) {
          println("Weight " + weiNum + ": " + neuron.m_outputWeights[weiNum].weight);
        }
      }
    }
  }
  
  //Render Neural Net
  void drawAll(int altY) {
    int cols = m_layers.length - 1;
    
    //render error
    fill(255);
    text("Fehlerrate: " + m_recentAverageError, 10, altY + 15);
    
    //render layers
    for(int c = 0; c <= cols; c++) {
      Layer l = m_layers[c];
      
      //Layer Header
      fill(255,255,0);
      textSize(26);
      text("L: " + c, startPosX + scale * (c + 1) * sizeScale - radius / 2, startPosY);
      
      //render neurons
      for(int r = 0; r < l.neuron.length; ++r) {
        Neuron n = l.neuron[r];
        
        //Connections / Lines
        if(c < cols) {
          for(int conn = 0; conn < m_layers[c+1].neuron.length - 1; conn++) {
            Neuron next = m_layers[c+1].neuron[conn];
            double weight = n.m_outputWeights[conn].weight;
            if(weight < 0) 
            { 
              stroke(0);
              strokeWeight(constrain((int)(weight * 10 * -1), 0, 2));
            } else {
              stroke(255);
              strokeWeight(constrain((int)(weight * 10), 0, 2));
            }
            
            line(n.x + radius, n.y, next.x - radius, next.y);
          }
        }
        
        //Neurons
        fill(constrain((int)(n.m_outputVal * 100), 0, 255));
        stroke(255);
        strokeWeight(1);
        ellipse(n.x, n.y, radius * 2, radius * 2);
        
        float textX = n.x - radius/1.3;
        float textY = n.y + radius/4;
        
        fill(255,255,0);
        textSize(11);
        text(nfs((float)n.m_outputVal, 0, 2) + "", textX, textY);
      }
    }
  }
  
  //FeedForward
  void feedForward(double[] inputVals) {
    //Assert that the inputValues are as many as neurons in the first layer
    assert(inputVals.length == m_layers[0].neuron.length - 1);

    //assign (latch) the input values into input neurons
    for (int i = 0; i < inputVals.length; ++i) {
      m_layers[0].neuron[i].setOutputVal(inputVals[i]);
    }
    
    //Forward propagate
    for (int layerNum = 1; layerNum < m_layers.length; ++layerNum) {
      Layer prevLayer = m_layers[layerNum - 1];
      for (int n = 0; n < m_layers[layerNum].neuron.length - 1; ++n) {
        m_layers[layerNum].neuron[n].feedForward(prevLayer);
      }
    }
  }
  
  //BackPropagation
  void backProp(double[] targetVals) {
    //Calculate overall net error (RMS of output neuron errors
    Layer outputLayer = m_layers[m_layers.length - 1];
    m_error = 0.0;

    for (int n = 0; n < outputLayer.neuron.length - 1; ++n) {
      double delta = targetVals[n] - outputLayer.neuron[n].getOutputVal();
      m_error += delta * delta;
    }
    m_error /= outputLayer.neuron.length - 1; //get average error squared
    m_error = Math.sqrt(m_error); //rms

    //Implement a revent average measurment:
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
  
  //Get the results
  double[] getResults(double[] resultVals) {
    resultVals = new double[] {};

    for (int n = 0; n < m_layers[m_layers.length - 1].neuron.length - 1; ++n) {
      resultVals = (double[])append(resultVals, m_layers[m_layers.length - 1].neuron[n].getOutputVal());
    }
    return resultVals;
  }
  
  
}