# Digit Recognizer with MLP

A simple machine learning project that uses a **Multi-Layer Perceptron (MLP)** to recognize handwritten digits (0–9). It includes a Python-based drawing interface and C++ for the core machine learning logic.

---

### Setup & Usage



### 1. Build the Project
To compile the C++ code:

```bash
make
```
This will create a main binary in the build/ folder specific to your system.

### 2. Run the App
Install the dependencies:

```bash
pip install -r requirements.txt
```
Then run the app:

```bash
python3 src/app.py
```
How to Use the App
Use your mouse to draw a digit (0–9) on the 28×28 pixel grid.

Click the "Predict" button.

The model will return its prediction at the top of the opened window.

### 3. Retraining the model (Optional)

> **Note:** Retraining the model is not necessary for demo purposes, functionality is there however.

If you still want to retrain on the MNIST number dataset run:

```bash
python3 src/train.py
```
This will download and preprocess the dataset, placing it into the training/ folder.

Then, open your C++ main.cpp file and:

Comment out the ```cpp predictionProcess(); ``` line

Uncomment the ```cpp trainingProcess(); ``` line

Recompile with:

```bash
make
```
Then to train run: 
```bash
./build/main
```
To train on another dataset, simply change what the train.py script downloads.

📋 Requirements
Python 3.8+  
For training, MNIST does not support anything above Python 3.9 I believe (For normal use ignore this).

C++ compiler

Virtualenv (recommended)

Install Python dependencies:

```bash
pip install -r requirements.txt
```
Why the Training Step Isn't Streamlined
The training process is kept separate and optional because most users do not need to retrain the model. The project is fully functional and demonstrable with the pre-trained version.

Author
GitHub: @tommatson

License
MIT License — feel free to use and modify.
