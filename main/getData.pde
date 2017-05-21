//get random inputData
double[] getNextInput() {
  double[] inputVals = new double[] { round(random(0, 1)), round(random(0, 1)), round(random(0, 1)), round(random(0, 1)) };
  return inputVals;
}

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