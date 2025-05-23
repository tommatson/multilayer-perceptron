🧠 Digit Recognizer with MLP
A simple machine learning project that uses a Multi-Layer Perceptron (MLP) to recognize handwritten digits (0–9). It includes a Python-based drawing interface and C++ for the core machine learning logic.

📦 Project Structure
.
├── src/ # Python and C++ source files
│ ├── train.py # Script to download and preprocess training data
│ ├── app.py # UI to draw and predict digits
├── training/ # Folder to hold training data
├── build/ # Compiled C++ binary ends up here
├── requirements.txt # Python dependencies
├── main.cpp # Main C++ logic
├── Makefile # Build system for C++ code
└── README.md # This file

🛠 Setup & Usage
1. 🧪 Prepare Training Data (Optional)
Note: Retraining the model is not necessary for demo purposes.

If you still want to retrain:

python3 src/train.py

This will download and preprocess the dataset, placing it into the training/ folder.

Then, open your C++ main.cpp file and:

Comment out the predictionProcess(); line

Uncomment the trainingProcess(); line

Recompile with:

make

2. ⚙️ Build the Project
To compile the C++ code:

make

This will create a main binary in the build/ folder.

3. 💻 Run the App
Install the dependencies:

pip install -r requirements.txt

Then run the app:

python3 src/app.py

🖼 How to Use the App
Use your mouse to draw a digit (0–9) on the 28×28 pixel grid.

Click the "Predict" button.

The model will return its prediction.

📸 Screenshots (Add Here)
Place images in a media/ folder and reference them like this:


Suggested screenshots:

Drawing interface in action

Example of a prediction

(Optional) Diagram of your MLP architecture

📋 Requirements
Python 3.8+

C++ compiler

Virtualenv (recommended)

Install Python dependencies:

pip install -r requirements.txt

🤔 Why the Training Step Isn't Streamlined
The training process is kept separate and optional because most users do not need to retrain the model. The project is fully functional and demonstrable with the pre-trained version.

💡 Possible Enhancements
Include a downloadable pre-trained model

Show model accuracy (e.g., on MNIST test set)

Add unit tests for Python components

Upgrade to a web interface (e.g., with Flask or Streamlit)

Add educational content explaining the MLP internals

🧑‍💻 Author
Your Name — yourwebsite.com
GitHub: @yourusername

📃 License
MIT License — feel free to use and modify.

