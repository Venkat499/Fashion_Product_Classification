Fashion Product Multi-Label Classification
This project uses a ResNet50 deep learning model to perform multi-label classification on fashion product images. Given an image, the model can identify its Product Type, Color, Season, and Gender. It includes a Streamlit web app for interactive demos and a Flask API for programmatic use.

Demo
Here are some sample predictions made by the application:

Uploaded Image

Predicted Color

Predicted Product Type

Predicted Season

Predicted Gender

<img src="https://raw.githubusercontent.com/Venkat499/Fashion_Product_Classification/main/Amazon%20sample/blue_men.png" width="150">

Blue

Tshirts

Summer

Men

<img src="https://raw.githubusercontent.com/Venkat499/Fashion_Product_Classification/main/Amazon%20sample/orange.png" width="150">

Orange

Kurtas

Fall

Women

<img src="https://raw.githubusercontent.com/Venkat499/Fashion_Product_Classification/main/Amazon%20sample/GREEN_women.png" width="150">

Green

Nightdress

Summer

Women

Features
Multi-Output Model: A single PyTorch model that predicts four different attributes.

Interactive Demo: A user-friendly Streamlit GUI to test the model with your own images.

REST API: A Flask endpoint for easy integration into other applications.

Complete Training Pipeline: A Jupyter Notebook is provided detailing the data preprocessing, model training, and evaluation.

Project Structure
.
├── API & Streamlit/
│   ├── api_inference.py        # Flask API for model inference
│   ├── streamlit_app.py        # Main Streamlit application file
│   ├── model.py                # Python script defining the model architecture
│   ├── le_colour.pkl           # Label encoder for color
│   ├── le_gender.pkl           # Label encoder for gender
│   ├── le_product_type.pkl     # Label encoder for product type
│   └── le_season.pkl           # Label encoder for season
│
├── Amazon sample/
│   └── ...                     # Sample images for testing
│
├── Jupyter File/
│   └── Fashion_Product_Classification.ipynb # Notebook with training code
│
├── best_model.pth              # Trained PyTorch model weights (Download from Releases)
└── README.md                   # This file

Getting Started
Follow these steps to set up and run the project on your local machine.

1. Prerequisites
Python 3.7+

pip and venv

2. Clone the Repository
git clone https://github.com/Venkat499/Fashion_Product_Classification.git
cd Fashion_Product_Classification

3. Set Up Virtual Environment & Install Dependencies
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install torch torchvision streamlit Pillow numpy scikit-learn flask

4. Download Model Weights
Download the best_model.pth file from the v1.0 Release on GitHub and place it in the root directory of the project.

Usage
You can interact with the model through the Streamlit app or the Flask API.

Option A: Run the Streamlit App (Interactive Demo)
This is the easiest way to test the project.

# Navigate to the app directory
cd "API & Streamlit"

# Run the app
streamlit run streamlit_app.py

Your browser will open to the application's interface.

Option B: Run the Flask API
Use the API for programmatic access.

# Navigate to the app directory
cd "API & Streamlit"

# Start the Flask server
python api_inference.py

The API will be available at http://127.0.0.1:5000. You can send a POST request with an image file to the /predict endpoint to get a JSON response.

Example curl request:

curl -X POST -F "file=@/path/to/your/image.jpg" http://127.0.0.1:5000/predict

Example JSON Output:

{
  "colour": "Blue",
  "gender": "Men",
  "product_type": "Tshirts",
  "season": "Summer"
}

Model Details
Architecture: ResNet50 (pretrained on ImageNet)

Framework: PyTorch

Image Size: 224x224 pixels

Data Augmentation: ColorJitter, Random Horizontal Flip, Random Rotation.

Dataset
This model was trained on the Fashion Product Images Dataset available on Kaggle, which contains over 44,000 labeled images.
