#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Constant definitions
#define INPUT_SIZE 784
#define HIDDEN1_SIZE 16
#define HIDDEN2_SIZE 16
#define OUTPUT_SIZE 10


// Define struct for storing network information
typedef struct {
    double input[INPUT_SIZE];
    double hidden1[HIDDEN1_SIZE];
    double hidden2[HIDDEN2_SIZE];
    double output[OUTPUT_SIZE];

    double weightsInputHidden1[INPUT_SIZE][HIDDEN1_SIZE];
    double weightsHidden1Hidden2[HIDDEN1_SIZE][HIDDEN2_SIZE];
    double weightsHidden2Output[HIDDEN2_SIZE][OUTPUT_SIZE];

    double biasHidden1[HIDDEN1_SIZE];
    double biasHidden2[HIDDEN2_SIZE];
    double biasOutput[OUTPUT_SIZE];

} NeuralNetwork;

// Functions initilise

void initialiseNetwork(NeuralNetwork *nn);

void backPropogate(NeuralNetwork *nn, int correctIndex, double learningRate){
    // error for each of the outputs
    double outputError[OUTPUT_SIZE];
    double hidden2Error[HIDDEN2_SIZE];
    double hidden1Error[HIDDEN1_SIZE];


    // find the error for the output layer by subtracting one from the intended number
    for (int i = 0; i < OUTPUT_SIZE; i++){
        outputError[i] = nn->output[i];
    }
    outputError[correctIndex] = outputError[correctIndex] - 1.0;

    // computer hidden layer 2 error
    for (int j =0; j < HIDDEN2_SIZE; j++){
        double sum = 0.0;
        for (int x = 0; x < OUTPUT_SIZE; x++) {
            sum += outputError[j] * nn->weightsHidden2Output[j][x];
        }
        hidden2Error[j] = ReLU(nn->hidden2[j]) * sum;
    }
    // computer hidden layer 1 error
    for (int k = 0; k < HIDDEN1_SIZE; k++){
        double sum = 0.0;
        for (int y = 0; y < HIDDEN2_SIZE; y++){
            sum+= hidden2Error[y] * nn->weightsHidden1Hidden2[k][y];
        }
        hidden1Error[k] = ReLU(nn->hidden1[k]) * sum;
    }
    


}

double ReLU(double x);

void forwardPass(NeuralNetwork *nn);

double crossEntropyLoss(double output[OUTPUT_SIZE], int correctIndex);


int main(void){
    // Initialse random function
    srand((unsigned int) time(NULL));

    // Initialise network function
    NeuralNetwork nn;
    initialiseNetwork(&nn);

    return 0;
}

void initialiseNetwork(NeuralNetwork *nn){
    printf("Initialising the network now...\n\n");
    // Increment over each input neuron and each hidden1 layer neuron to assign a random weight
    for(int x = 0; x< INPUT_SIZE; x++){
        for (int y = 0; y < HIDDEN1_SIZE; y++){
            // Assign the struct value with nn-> then  calculate a random number
            // divide my max to make it between 0 - 1 then multiply by 2 to make it beween 0 - 2
            // then subtract 1 to make it between (-1) - 1
            nn->weightsInputHidden1[x][y] = ((double) rand() / RAND_MAX) * 2.0 - 1.0;
        }

    }

    // Hidden layer 1 and hidden layer 2
    for (int x= 0; x < HIDDEN1_SIZE; x++){
        for (int y = 0; y < HIDDEN2_SIZE; y++){
            nn->weightsHidden1Hidden2[x][y] = ((double) rand() / RAND_MAX) * 2.0 - 1.0;
        }
    }

    // Hidden layer 2 and output layer
    for (int x= 0; x < HIDDEN2_SIZE; x++){
        for (int y = 0; y < OUTPUT_SIZE; y++){
            nn->weightsHidden2Output[x][y] = ((double) rand() / RAND_MAX) * 2.0 - 1.0;
        }
    }

    // Set bias to zero
    for (int x = 0; x < HIDDEN1_SIZE; x++){
        nn->biasHidden1[x] = 0.0;
    }
    for (int x = 0; x < HIDDEN2_SIZE; x++){
        nn->biasHidden2[x] = 0.0;
    }
    for (int x = 0; x < OUTPUT_SIZE; x++){
        nn->biasOutput[x] = 0.0;
    }
    printf("Network has been initilised successfully yay!\n");
}

double ReLU(double x){
    // If X is less then zero return 0
    return (x > 0) ? x : 0.0;
}

void forwardPass(NeuralNetwork *nn){
    // Used to set the activations of the neurons

    // Hidden layer 1 - ReLU
    for (int x = 0; x < HIDDEN1_SIZE; x++) {
        double sum = 0.0;
        for (int y = 0; y < INPUT_SIZE; y++){
            sum += nn->input[y] * nn->weightsInputHidden1[y][x];
        }
        nn->hidden1[x] = ReLU(sum + nn->biasHidden1[x]);
    }

    // Hidden layer 2 - ReLU
    for (int x = 0; x < HIDDEN2_SIZE; x++) {
        double sum = 0.0;
        for (int y = 0; y < HIDDEN1_SIZE; y++){
            sum += nn->hidden1[y] * nn->weightsHidden1Hidden2[y][x];
        }
        nn->hidden2[x] = ReLU(sum + nn->biasHidden2[x]);
    }

    // Output layer - Softmax
    double secondSum = 0.0;
    for (int x = 0; x < OUTPUT_SIZE; x++){
        double firstSum = 0.0;
    
        for (int y = 0; y < HIDDEN2_SIZE; y++){
            firstSum += nn->hidden2[y] * nn->weightsHidden2Output[y][x];
        }
        nn->output[x] = exp(firstSum + nn->biasOutput[x]);
        secondSum += nn->output[x];
    }
    for (int x = 0; x < OUTPUT_SIZE; x++){
        nn->output[x] /= secondSum;
    }

}

double crossEntropyLoss(double output[OUTPUT_SIZE], int correctIndex){
    // Loss function works by finding the correct output stored as a 1D array e.g [0, 1, 0, 0, 0, 0, 0, 0, 0, 0] for the number 1
    // It then uses the log () of the found probability found via the softmax function and multiplies this by 1
    // Output stores all the values of the output neurons and correctIndex stores the meant to be index of the value
    return -log(output[correctIndex]);

}