//GLOBAL VARIABLES
//Make Neural Net
Net myNet;
//Create variables
double[] inputVals, targetVals, resultVals;
//Update variable
boolean update = false;
//Trainings counter
int trainingPass = 0;

//Accept rate for rendering
float renderRate = 0.05;

//Setup functions
void setup() {
  //Set window size
  size(900, 900);
  
  //Create topology -> Item = Neurons in this layer
  int[] topology = new int[] { 4, 5, 6, 5};
  //Pass topology to Net
  myNet = new Net(topology);
  
  //Set Variables
  inputVals = new double[] {};
  targetVals = new double[] {};
  resultVals = new double[] {};
}

//Draw function
void draw() {
  //set background
  background(51);
  
  //true = train net
  if(update){
    //********************GET DATA************************
    inputVals = getNextInput();
    targetVals = calcOutput(inputVals);
    //****************************************************
    
    //Update net
    myNet.feedForward(inputVals);
    //get the results
    resultVals = myNet.getResults(resultVals);
    //Backpropagation
    myNet.backProp(targetVals);
    
    //Training counter
    trainingPass++;
  }
  
  //********************Extra info************************
  int y = 20;
  int x = 10;
  
  //Show Input Values
  for (int n = 0; n < inputVals.length; ++n) {
    fill(255, 100, 0);
    text("Inputvalue " + n + ": " + inputVals[n] + "", x, y + 15 * (n+1));
  }
  //Show Target Values
  y += 15 * inputVals.length;
  for (int n = 0; n < targetVals.length; ++n) {
    fill(0, 255, 255);
    text("Targetvalue " + n + ": " + targetVals[n], x, y + 15 * (n+1));
  }
  //Show Result Values
  y += 15 * targetVals.length;
  for (int n = 0; n < resultVals.length; ++n) {
    //If the Value is in range of 5% set its color green
    if(abs((float)(targetVals[n] - resultVals[n])) < renderRate) {
      fill(0, 255, 0);
    } else {
      fill(255, 0, 0);
    }
    text("Result " + n + ": " + resultVals[n] + "", x, y + 15 * (n+1));
  }
  //Show trainings count
  y += 15 * resultVals.length;
  fill(255);
  text(trainingPass + " runs", x, y + 15); 
  //****************************************************
  
  //Call the drawfunction to access the Net data
  myNet.drawAll(y + 15);
  
  
  //######################################################################################################
  //RENDER INPUT
  int scale = 40;
  for(int c = 0; c < inputVals.length; c++) {
    if(inputVals[c] == 0) {
      fill(0);
    } else {
      fill(255);
    }
    noStroke();
    rect((c+1)* scale, y + 45, scale, scale);
  }
}

//MouseClick function
void mouseClicked() {
  //Toggle training
  update = (update == true) ? false : true;
}

//keyPressed function
void keyPressed() {
  //print Weights
  if (key == 'w' && update == false) {
    myNet.printWeights();
  }
}