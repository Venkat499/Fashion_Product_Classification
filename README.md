# Fashion Product Multi-Label Classification

## Overview
This project demonstrates multi-label classification of fashion product images using deep learning (ResNet50).  
The model predicts:
- **Color**
- **Product Type** (e.g., Tshirts, Shoes, etc.)
- **Season**
- **Gender**

## Features
- PyTorch-based multi-output model
- Trained on the Kaggle Fashion Product Images Dataset
- FastAPI endpoint for inference
- Streamlit GUI for demo

## Project Structure
```
CodeMonk_Task/
├── Fashion_Product_Classification.ipynb   # Main notebook (EDA, training, evaluation)
├── model.py                              # Model definition
├── best_model.pth                        # Trained model weights
├── le_colour.pkl                         # Color label encoder
├── le_product_type.pkl                   # Product type label encoder
├── le_season.pkl                         # Season label encoder
├── le_gender.pkl                         # Gender label encoder
├── api_inference.py                      # FastAPI server
├── streamlit_app.py                      # Streamlit GUI
├── requirements.txt                      # Python dependencies
├── amazon_samples/                       # (Optional) Sample images for testing
└── screenshots/                          # Demo screenshots
```

## Setup Instructions

1. **Clone the repository**
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Download the trained model and encoders**
   - Download `best_model.pth`, `le_colour.pkl`, `le_product_type.pkl`, `le_season.pkl`, `le_gender.pkl` from the [GitHub release](YOUR_RELEASE_LINK).
   - Place them in the project root directory.

4. **Run the notebook**
   - Open `Fashion_Product_Classification.ipynb` in Jupyter/Kaggle/Colab
   - Run all cells in order

5. **Run the API**
   ```bash
   python api_inference.py
   # Visit http://localhost:8000/docs
   ```

6. **Run the Streamlit app**
   ```bash
   streamlit run streamlit_app.py
   ```

## Usage

- **API:** Upload an image to `/predict/` endpoint to get predictions.
- **Streamlit:** Use the web interface to upload images and view predictions.

## Model Details

- **Backbone:** ResNet50 (pretrained on ImageNet)
- **Image Size:** 224x224
- **Augmentation:** ColorJitter, RandomFlip, RandomRotation
- **Loss:** Weighted CrossEntropy (higher weight for color)
- **Framework:** PyTorch

## Dataset

- [Kaggle Fashion Product Images Dataset](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset)
- ~44,000 images with labels for color, product type, season, and gender

## Demo Results

Below are sample predictions from the Streamlit app using real Amazon fashion product images:

| Uploaded Image | Predicted Color | Predicted Product Type | Predicted Season | Predicted Gender |
|:--------------:|:---------------:|:---------------------:|:----------------:|:----------------:|
| ![Blue Tshirt](screenshots/blue_tshirt.png) | Blue | Tshirts | Summer | Men |
| ![Orange Kurta](screenshots/orange_kurta.png) | Orange | Kurtas | Fall | Women |
| ![Green Dress](screenshots/green_dress.png) | Green | Nightdress | Summer | Women |

**JSON Output Example:**
```json
{
  "colour": "Blue",
  "product_type": "Tshirts",
  "season": "Summer",
  "gender": "Men"
}
```

**Summary Output Example:**
- 🎨 Color: Blue
- 👕 Product Type: Tshirts
- 🌤️ Season: Summer
- 👤 Gender: Men

## Model Performance

- **Color Accuracy:** 80%+
- **Product Type Accuracy:** 85%+
- **Season Accuracy:** 75%+
- **Gender Accuracy:** 90%+

## Author
[Your Name] - Codemonk ML Intern Assignment 