// Doing the thing again in C++

#include <iostream>
#include <random>


#define INPUT_SIZE 784
#define OUTPUT_SIZE 10
#define HIDDEN1_SIZE 16
#define HIDDEN2_SIZE 16

struct Network {
    double input[INPUT_SIZE];
    double output[OUTPUT_SIZE];

    double hidden1[HIDDEN1_SIZE];
    double hidden2[HIDDEN2_SIZE];

    // Weights between input layer and output layer
    double weights1[HIDDEN1_SIZE][INPUT_SIZE];
    // Biases for each neuron on the output layer
    double biases1[HIDDEN1_SIZE];

    double weights2[HIDDEN2_SIZE][HIDDEN1_SIZE];
    double biases2[HIDDEN2_SIZE];

    double weightsOutput[OUTPUT_SIZE][HIDDEN2_SIZE];
    double biasesOutput[OUTPUT_SIZE];
};


double applySigmoid(double value){
    // :) don't have to program taylor series for e^x, blessed is the machine spirit
    return 1.0 / (1.0 + exp(-value));
}

void forwardPass(Network* myNetwork){
    // Forward pass, usual logic of sigmoid(weights * activation + bias) for a given neuron
    int sumWeightsActivation = 0;
    for(int i = 0; i < HIDDEN1_SIZE; i++){
        sumWeightsActivation = 0;
        for(int j = 0; j < INPUT_SIZE; j++){
            sumWeightsActivation += myNetwork->weights1[i][j] * myNetwork->input[j];
        }
        myNetwork->hidden1[i] = applySigmoid(sumWeightsActivation + myNetwork->biases1[i]);
    }

    for(int i = 0; i < HIDDEN2_SIZE; i++){
        sumWeightsActivation = 0;
        for(int j = 0; j < HIDDEN1_SIZE; j++){
            sumWeightsActivation += myNetwork->weights2[i][j] * myNetwork->hidden1[j];
        }
        myNetwork->hidden2[i] = applySigmoid(sumWeightsActivation + myNetwork->biases2[i]);
    }

    for(int i = 0; i < OUTPUT_SIZE; i++){
        sumWeightsActivation = 0;
        for(int j = 0; j < HIDDEN2_SIZE; j++){
            sumWeightsActivation += myNetwork->weightsOutput[i][j] * myNetwork->hidden2[j];
        }
        myNetwork->output[i] = applySigmoid(sumWeightsActivation + myNetwork->biasesOutput[i]);
    }    
}


void initialiseNetwork(Network* myNetwork){
    // Initalise the network with random weights and biases

    // Get a random number from hardware, we use this as a seed for our random number generator
    std::random_device rd;
    std::mt19937 gen(rd());

    // Using the Xavier method for weight initalisation as this pairs well with sigmoid
    // (basically create a normal distribution with mean 0, and variance based on input / output size, then pick a random number from it)
    std::normal_distribution<float> dist(0.0, sqrt(2.0 / (INPUT_SIZE + OUTPUT_SIZE)));

    for(int i = 0; i < HIDDEN1_SIZE; i++){
        for(int j = 0; j < INPUT_SIZE; j++){
            // Generate a random weight using our distro
            myNetwork->weights1[i][j] = dist(gen);
        }
    }
    // Do this for all the weights
    for(int i = 0; i < HIDDEN2_SIZE; i++){
        for(int j = 0; j < HIDDEN1_SIZE; j++){
            myNetwork->weights2[i][j] = dist(gen);
        }
    }
    for(int i = 0; i < OUTPUT_SIZE; i++){
        for(int j = 0; j < HIDDEN2_SIZE; j++){
            myNetwork->weightsOutput[i][j] = dist(gen);
        }
    }
    // Zero initialise the biases
    for(int i = 0; i < HIDDEN1_SIZE; i++){
        myNetwork->biases1[i] = 0;
    }
    for(int i = 0; i < HIDDEN2_SIZE; i++){
        myNetwork->biases2[i] = 0;
    }
    for(int i = 0; i < OUTPUT_SIZE; i++){
        myNetwork->biasesOutput[i] = 0;
    }
  
  

}


int main(){
    Network* myNetwork = new Network();
    initialiseNetwork(myNetwork);
    return 0;
}

