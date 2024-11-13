from fastapi import FastAPI, UploadFile, File
from PIL import Image
import torch
from torchvision import models, transforms
import io
import requests
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Настройки CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500","*"],  # Разрешить запросы только с этого источника
    allow_credentials=True,
    allow_methods=["*"],  # Разрешить все методы (POST, GET и т.д.)
    allow_headers=["*"],  # Разрешить любые заголовки
)
# Загрузка предобученной модели ResNet
model = models.resnet50(pretrained=True)
model.eval()  # Перевод модели в режим оценки

# Загрузка классов ImageNet
LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
labels = requests.get(LABELS_URL).json()

# Подготовка трансформаций для изображения
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Функция для предсказания класса изображения
def predict_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image_tensor = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image_tensor)
    _, predicted_idx = torch.max(output, 1)

    return labels[predicted_idx.item()]

# Маршрут для загрузки изображения и предсказания
@app.post("/predict/")
async def categorize_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    prediction = predict_image(image_bytes)
    return {"category": prediction}

# Запуск сервера Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
