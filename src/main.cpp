// Doing the thing again in C++

#include <iostream>
#include <random>
#include <fstream>
#include <vector>
#include <sstream>


// TO DO :
// 1. TRAIN model
// 2. Python build the interface

#define INPUT_SIZE 784
#define OUTPUT_SIZE 10
#define HIDDEN1_SIZE 16
#define HIDDEN2_SIZE 16


// For gradient descent
#define LEARNING_RATE 0.01


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


    double errorOutput[OUTPUT_SIZE];
    double errorHidden1[HIDDEN1_SIZE];
    double errorHidden2[HIDDEN2_SIZE];

    // Will be either 0 or 1 (double because of how padding works in structs)
    double actual[OUTPUT_SIZE];
};

void saveNetwork(Network* myNetwork, std::string filename){
    // Create the output file
    std::ofstream outFile(filename, std::ios::binary);
    if (outFile){
        // Write the struct to the output file, casting the network pointer to a char since this is what the write expects
        outFile.write(reinterpret_cast<const char*>(myNetwork), sizeof(Network));
    }
    outFile.close();

}
Network* readNetwork(std::string filename){
    Network* myNetwork = new Network();
    std::ifstream inFile(filename, std::ios::binary);
    if (inFile) {
        inFile.read(reinterpret_cast<char*>(myNetwork), sizeof(Network));
    }
    inFile.close();
    return myNetwork;
}

void calculateErrorTerms(Network* myNetwork){
    // Using the formula for the output layer error term which we derive from the cost function of 1/2 (y1 - y0)^2 where y1 is predicted and y0 is actual
    for(int i = 0; i < OUTPUT_SIZE; i++){
        myNetwork->errorOutput[i] = (myNetwork->output[i] - myNetwork->actual[i]) * (myNetwork->output[i]) * (1 - myNetwork->output[i]);
    }
    // Var error is used as this is more efficient than dereferencing the pointer each j loop iteration
    // Like error, activation is used to reduce need for dereferencing pointers

    // Now we have the error for the output layer we can use this for the next layer
    for (int i = 0; i < HIDDEN2_SIZE; i++){
        double error = 0;
        double activation = myNetwork->hidden2[i];

        for (int j = 0; j < OUTPUT_SIZE; j++){
            // Use the formula of the weights * error * derivative (derivative is then f(z)(1 - f(z)) where f(z) is our activation)
            error += myNetwork->weightsOutput[j][i] * myNetwork->errorOutput[j] * (1 - activation) * activation;
        }
        myNetwork->errorHidden2[i] = error;
    }

    // Repeat the exact same thing for the other layer
    for (int i = 0; i < HIDDEN1_SIZE; i++){
        double error = 0;
        double activation = myNetwork->hidden1[i];
        for (int j = 0; j < HIDDEN2_SIZE; j++){
            // Use the formula of the weights * error * derivative (derivative is then f(z)(1 - f(z)) where f(z) is our activation)
            error += myNetwork->weights2[j][i] * myNetwork->errorHidden2[j] * (1 - activation) * activation;
        }
        myNetwork->errorHidden1[i] = error;
    }

}

void gradientDescent(Network* myNetwork){


    // Update the weights and biases using gradient descent
    for (int i = 0; i < HIDDEN1_SIZE; i++){
        double error = myNetwork->errorHidden1[i];
        for(int j = 0; j < INPUT_SIZE; j++){
            myNetwork->weights1[i][j] -= LEARNING_RATE * error * myNetwork->input[j];

        }
        myNetwork->biases1[i] -= LEARNING_RATE * error;
    }

    for (int i = 0; i < HIDDEN2_SIZE; i++){
        double error = myNetwork->errorHidden2[i];
        for(int j = 0; j < HIDDEN1_SIZE; j++){
            myNetwork->weights2[i][j] -= LEARNING_RATE * error * myNetwork->hidden1[j];

        }
        myNetwork->biases2[i] -= LEARNING_RATE * error;
    }

    for (int i = 0; i < OUTPUT_SIZE; i++){
        double error = myNetwork->errorOutput[i];
        for(int j = 0; j < HIDDEN2_SIZE; j++){
            myNetwork->weightsOutput[i][j] -= LEARNING_RATE * error * myNetwork->hidden2[j];

        }
        myNetwork->biasesOutput[i] -= LEARNING_RATE * error;
    }

}

void backPropagate(Network* myNetwork){
    calculateErrorTerms(myNetwork);
    gradientDescent(myNetwork);
}


double applySigmoid(double value){
    // :) don't have to program taylor series for e^x, blessed is the machine spirit
    return 1.0 / (1.0 + exp(-value));
}

void forwardPass(Network* myNetwork){
    // Forward pass, usual logic of sigmoid(weights * activation + bias) for a given neuron
    double sumWeightsActivation = 0.0;
    for(int i = 0; i < HIDDEN1_SIZE; i++){
        sumWeightsActivation = 0.0;
        for(int j = 0; j < INPUT_SIZE; j++){
            sumWeightsActivation += myNetwork->weights1[i][j] * myNetwork->input[j];
        }
        myNetwork->hidden1[i] = applySigmoid(sumWeightsActivation + myNetwork->biases1[i]);
    }

    for(int i = 0; i < HIDDEN2_SIZE; i++){
        sumWeightsActivation = 0.0;
        for(int j = 0; j < HIDDEN1_SIZE; j++){
            sumWeightsActivation += myNetwork->weights2[i][j] * myNetwork->hidden1[j];
        }
        myNetwork->hidden2[i] = applySigmoid(sumWeightsActivation + myNetwork->biases2[i]);
    }

    for(int i = 0; i < OUTPUT_SIZE; i++){
        sumWeightsActivation = 0.0;
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

    // for weights1
    std::normal_distribution<float> dist1(0.0, sqrt(1.0 / INPUT_SIZE));
    // for weights2
    std::normal_distribution<float> dist2(0.0, sqrt(1.0 / HIDDEN1_SIZE));
    // for weightsOutput
    std::normal_distribution<float> distOut(0.0, sqrt(1.0 / HIDDEN2_SIZE));

    for(int i = 0; i < HIDDEN1_SIZE; i++){
        for(int j = 0; j < INPUT_SIZE; j++){
            // Generate a random weight using our distro
            myNetwork->weights1[i][j] = dist1(gen);
        }
    }
    // Do this for all the weights
    for(int i = 0; i < HIDDEN2_SIZE; i++){
        for(int j = 0; j < HIDDEN1_SIZE; j++){
            myNetwork->weights2[i][j] = dist2(gen);
        }
    }
    for(int i = 0; i < OUTPUT_SIZE; i++){
        for(int j = 0; j < HIDDEN2_SIZE; j++){
            myNetwork->weightsOutput[i][j] = distOut(gen);
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



std::vector<std::vector<int>> loadBinaryCSV(std::string filename){
    std::vector<std::vector<int>> data;
    std::ifstream file(filename);
    std::string line;
    while (std::getline(file, line)){
        std::stringstream ss(line);
        std::string value;
        std::vector<int> row;

        while (std::getline(ss, value, ',')) {
            row.push_back(std::stoi(value));
        }

        data.push_back(row);
    }
    // for(int i = 0; i < data.size(); i++){
    //     for (int j = 0; j < data[i].size(); j++){
    //         std::cout << i << "/" << j << std::endl;
    //     }
    // }
    return data;
}

std::vector<int> loadLabelsCSV(std::string filename){
    std::vector<int> data;
    std::ifstream file(filename);
    std::string line;
    while (std::getline(file, line)) {
        data.push_back(std::stoi(line));
    }
    return data;
}



void trainNetwork(Network* myNetwork, std::string inputFilename, std::string labelFilename){
    std::vector<std::vector<int>> inputData = loadBinaryCSV(inputFilename);
    std::vector<int> labelData = loadLabelsCSV(labelFilename);
    int epochs = 10;
    // Train the model on the input data
    for (int k = 0; k < epochs; k++){
        for (int i = 0; i < inputData.size(); i++){
            // Set up all the input
            for(int j = 0; j < INPUT_SIZE; j++){
                myNetwork->input[j] = inputData[i][j];
            }
            // Set up the actual
            for(int j = 0; j < OUTPUT_SIZE; j++){
                myNetwork->actual[j] = labelData[i] == j ? 1.0 : 0.0;
            }
            // Usual network steps now that the input and actual are set
            forwardPass(myNetwork);
            // double loss = 0.0;
            // for (int j = 0; j < OUTPUT_SIZE; j++) {
            //     double diff = myNetwork->output[j] - myNetwork->actual[j];
            //     loss += diff * diff;
            // }
            // std::cout << "Sample " << i << " loss: " << loss << std::endl;
            backPropagate(myNetwork);
        }
    }
    
}


// Used to print the value and weights of a neuron for testing purposes
void printNeuron(Network* myNetwork){
    std::cout << "Activation: " << myNetwork->hidden1[0] << std::endl;
    for (int i = 0; i < HIDDEN2_SIZE; i++){
        std::cout << "Weight: " << myNetwork->weights2[i][1] << std::endl; 
    }
}




std::vector<int> loadInputTxt(std::string filename){
    std::ifstream infile(filename);
    std::vector<int> values;
    int val;
    
    if (!infile) {
        std::cerr << "Error opening input.txt\n";
    } 
    
    // Read all integers from the file
    while (infile >> val) {
        values.push_back(val);
    }
    return values;

}


void trainingProcess(){
    Network* myNetwork = new Network();
    initialiseNetwork(myNetwork);
    trainNetwork(myNetwork, "training/mnist_binary_input.csv", "training/mnist_labels.csv");
    saveNetwork(myNetwork, "data/network1");
}

void predictionProcess(){
    Network* myNetwork = readNetwork("data/network1");
    std::vector<int> inputData = loadInputTxt("data/input.txt");
    for(int i = 0; i < INPUT_SIZE; i++){
        myNetwork->input[i] = inputData[i];
    }
    forwardPass(myNetwork);

    // for (int i = 0; i < INPUT_SIZE; i++){
    //     std::cout << i << ":" << myNetwork->weights1[0][i] << std::endl;
    // // }
    // std::cout << "Activation" << myNetwork->hidden1[0] <<  "Bias" << myNetwork->biases1[0] << std::endl;
    // Find the max output
    double max = 0.0;
    int maxIndex = 0;
    for (int i = 0; i < OUTPUT_SIZE; i++){
        // std::cout << myNetwork->output[i] << std::endl; 
        if (myNetwork->output[i] > max){
            max = myNetwork->output[i];
            maxIndex = i;
        }
    }
    // Write the max
    std::ofstream outfile("data/output.txt");
    if (!outfile) {
        std::cerr << "Error opening output.txt for writing\n";
    }

    outfile << maxIndex; // Write the int to file
    outfile.close();

}

int main(){

    // trainingProcess();
    predictionProcess();
    return 0;
}

