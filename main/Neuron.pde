//Neuron Class
class Neuron {
  //Varialbes
  double eta = 0.15;    //[0.0..1.0] overall net training rate
  double alpha = 0.5;  //[0.0..n] multiplier of last weight change (momentum)
  double m_outputVal;
  Connection[] m_outputWeights = new Connection[] {};
  int m_myIndex;
  double m_gradient;
  
  //Render Variables
  float x;
  float y;
  
  //Neuron Constructor
  Neuron(int numOutputs, int myIndex) {
    for (int c = 0; c < numOutputs; ++c) {
      //Add a neuron
      m_outputWeights = (Connection[])append(m_outputWeights, new Connection());
      //++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
      //assign random weight / or set the weight
      m_outputWeights[m_outputWeights.length - 1].weight = randomWeight();
    }
    
    //set the index
    m_myIndex = myIndex;
  }
  
  //Set the output value
  void setOutputVal(double val) { 
    m_outputVal = val; 
  }
  
  //Get the output value
  double getOutputVal() { 
    return m_outputVal; 
  }
  
  //update the input weight
  void updateInputWights(Layer prevLayer) {
    //the weight to  be updated are in the Connection container
    //in the neurons in the preceding layer
    for (int n = 0; n < prevLayer.neuron.length; ++n) {
      Neuron neuron = prevLayer.neuron[n];
      double oldDeltaWeight = neuron.m_outputWeights[m_myIndex].deltaWeight;

      double newDeltaWeight =
        //Individual input, magnified by the gradient and train rate:
        eta
        * neuron.getOutputVal()
        * m_gradient
        //Also add momentum = a fraction of the previous delta weight
        + alpha
        * oldDeltaWeight;

      neuron.m_outputWeights[m_myIndex].deltaWeight = newDeltaWeight;
      neuron.m_outputWeights[m_myIndex].weight += newDeltaWeight;
    }
  }
  
  //calculate the hidden gradients
  void calcHiddenGradients(Layer nextLayer) {
    double dow = sumDOW(nextLayer);
    m_gradient = dow * transferFunctionDerivative(m_outputVal);
  }
  
  //calculate the output gradients
  void calcOutputGradients(double targetVal) {
    double delta = targetVal - m_outputVal;
    m_gradient = delta * transferFunctionDerivative(m_outputVal);
  }
  
  //feedForward
  void feedForward(Layer prevLayer) {
    double sum = 0.0;
    //Sum the precious layer´s outputs (which are our inputs)
    //Include the bias node from the previous layer.

    for (int n = 0; n < prevLayer.neuron.length; ++n) {
      sum += prevLayer.neuron[n].getOutputVal() 
        * prevLayer.neuron[n].m_outputWeights[m_myIndex].weight;
    }

    m_outputVal = transferFunction(sum);
  }
  
  //transferFunc
  double transferFunction(double x) {
    //tanh - output range [-1.0 .. 1.0]
    return Math.tanh(x);
  }
  
  //DerivateTransferFunc
  double transferFunctionDerivative(double x) {
    //tanh derivative
    return 1.0 - x * x;
  }
  
  //SumDOW
  double sumDOW(Layer nextLayer) {
    double sum = 0.0;

    //sum our contributions of the errors at the nodes we feed

    for (int n = 0; n < nextLayer.neuron.length - 1; ++n) {
      sum += m_outputWeights[n].weight * nextLayer.neuron[n].m_gradient;
    }

    return sum;
  }
  
  //random weight function
  double randomWeight() {
    return random(1000) / 1000; 
  }
}