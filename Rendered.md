# Rendered

<img src="/images/MainFrame.png"></img>

### Explanation
There are a few things to explain here.  

First the different texts on the left side. These are the values for the neural  net.  
We have 4 input values in orange at the top left corner. These are the values we give to the input layer of the neural net. These change 
for each drawloop. Underneath in the bluish color we have the target values. These are the values the neural net should calculate. These 
values change for every drawloop aswell. Under it we find the actual values the neural net calculates. Green means that the values is in the
5% error rate. Red means that the value is not. These change, who would have guess, as well.
```processing
 //We render these lines here
 //main.pde line 68
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
```

In white we have the total number of iterations and the total Error rate.
```processing
    //render error
    //Net.pde line 70
    fill(255);
    text("Error rate: " + m_recentAverageError, 10, altY + 15);
```

And underneath the whole text we render the input values to visualize these.

Now we come to the right site of the window.  
Here we have the actual neural net. Each column is a layer. These get labeled with **L: n**  
Each circle is a neuron. It has a number in it that indicates the output value the neuron has. From each single neuron that is not in the
last layer goes a connection to each neuron in the next layer. These are the lines between the neurons. White means that the weight of the
connection is positive (greater than zero) and black means that it is negative (lower than zero). The thicker the line is the higher the
weight.  
You will notice that there will always be a neuron with a **1.0** at the bottom of each layer. These are the bias neurons. They are
necessary for the calculation. These influence the neurons in the next layer but don´t get influenced by the ones in the layer before.
