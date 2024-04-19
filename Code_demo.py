import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F

#Archicture
class model_cnn(nn.Module):
    def __init__(self, num_classes=10): 
        super(model_cnn, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(48, 96, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(96, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(192 * 7 * 7, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Loading the model
model = model_cnn()
model.load_state_dict(torch.load('cnn_model2.pth', map_location=torch.device('cpu')))
model.eval()

classes = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

#Transforming images
def transform_image(image_file):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    image = Image.open(image_file).convert('RGB')
    return transform(image).unsqueeze(0)

# Streamlit web interface
st.markdown("<h1 style='text-align: center; color: white;'>Interactive CNN Image Classifier</h1>", unsafe_allow_html=True)
uploads = st.file_uploader("Choose images:", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploads:
    cols = st.columns(len(uploads))  
    for uploaded_file, col in zip(uploads, cols):
        with col:  
            image = transform_image(uploaded_file)
            st.image(uploaded_file, caption='Uploaded Image', width=80)

            if st.button('Predict', key=uploaded_file.name): 
                output = model(image)
                probs = F.softmax(output, dim=1)
                pred_class = torch.argmax(probs, dim=1)
                class_name = classes[pred_class.item()]
                st.write(f'Predicted Class: {class_name}')
