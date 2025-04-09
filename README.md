# MoE Layer Interactive Demo

This project demonstrates a Mixture of Experts (MoE) layer implemented with PyTorch and integrated into a Flask web application. The demo allows users to input a 10-element vector, which is processed by the MoE layer. Only the top 2 experts (out of 4) are activated per input, offering dynamic computation graph construction and compute savings. The web interface provides real-time visualization of expert selection probabilities and output tensor values.

## Features

- **Dynamic Expert Selection:** Only 2 out of 4 experts are activated for each input, reducing computation by approximately 50%.
- **Gradient Isolation:** Only the activated experts receive gradients, which helps isolate learning signals.
- **Real-Time Visualization:** A user-friendly HTML interface displays expert selection probabilities and the output tensor.
- **Multiple Examples:** Built-in examples demonstrate both linear and non-linear input patterns.
- **Performance Metrics:** Compare theoretical performance metrics between a Dense Layer and the MoE Layer.

## Requirements

- Python 3.7+
- [PyTorch](https://pytorch.org/)
- [Flask](https://flask.palletsprojects.com/)

## Installation

1. **Clone the repository:**

   git clone github.com/StephanFWard/AI-Mixture-of-Expert-Optimization
   cd AI-Mixture-of-Expert-Optimization

  Create and activate a virtual environment (optional but recommended):

## Environment Setup (Optional
python -m venv venv
source venv/bin/activate   # On Windows, 
use: venv\Scripts\activate

2. **Install the required packages:**

  pip install torch flask
  
Running the Application
Ensure app.py is available and the index.html is in the templates folder

3. **Start the Flask application:**
  python app.py

4. Open your web browser and navigate to http://127.0.0.1:5000 to interact with the MoE demo.

**Project Structure**

Edit
├── app.py         # Flask application integrating the PyTorch MoE layer
├── templates/index.html     # HTML file for the interactive demo interface
└── README.md      # Project documentation file

**Technical Details**

  - MoE Layer: Implemented as a PyTorch nn.Module that:

  - Computes gating probabilities using a softmax on a gating network.

  - Dynamically selects the top 2 experts based on the gating probabilities.

-  - Aggregates the experts' outputs weighted by their corresponding probabilities.

**Web Interface: Built with Flask to allow users to:**

  1. Input a comma-separated list of 10 values.

  2. View real-time results with clearly labeled output tensor and expert selection probabilities.

  3.  Performance Metrics: The demo provides insight into computational benefits by comparing FLOPs, memory usage, and training time of MoE Layers vs. Dense Layers.

**Additional Notes:**

  The project includes inline code comments for clarity.

  Example inputs are provided within the interface for ease of testing.

  This is a prototype to demonstrate the concepts behind dynamic expert selection and conditional computation in neural networks.
