import torch
import pickle
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io
import torchvision.transforms as transforms
from model import MultiOutputModel

import warnings
warnings.filterwarnings('ignore')

# Load encoders
with open('le_colour.pkl', 'rb') as f: le_colour = pickle.load(f)
with open('le_product_type.pkl', 'rb') as f: le_product_type = pickle.load(f)
with open('le_season.pkl', 'rb') as f: le_season = pickle.load(f)
with open('le_gender.pkl', 'rb') as f: le_gender = pickle.load(f)

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MultiOutputModel(
    n_colours=len(le_colour.classes_),
    n_product_types=len(le_product_type.classes_),
    n_seasons=len(le_season.classes_),
    n_genders=len(le_gender.classes_)
)
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model.eval()

app = FastAPI()

IMG_SIZE = 224
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    img = transform(img).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img)
        preds = {
            'colour': le_colour.inverse_transform([outputs['colour'].argmax(1).item()])[0],
            'product_type': le_product_type.inverse_transform([outputs['product_type'].argmax(1).item()])[0],
            'season': le_season.inverse_transform([outputs['season'].argmax(1).item()])[0],
            'gender': le_gender.inverse_transform([outputs['gender'].argmax(1).item()])[0]
        }
    return preds

@app.post("/predict/")
async def predict_api(file: UploadFile = File(...)):
    img_bytes = await file.read()
    preds = predict(img_bytes)
    return preds

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 