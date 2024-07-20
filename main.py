import streamlit as st
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import time
import numpy as np
from PIL import Image
import sys
import pandas as pd

from torchvision import models

class MultimodalModel(nn.Module):
    def __init__(self, num_landmarks, num_classes):
        super(MultimodalModel, self).__init__()
        # Image processing branch using ResNet18
        self.image_branch = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        num_ftrs = self.image_branch.classifier[1].in_features
        #self.image_branch.fc = nn.Identity()  # Remove the final classification layer
        self.image_branch.classifier[1] = nn.Linear(num_ftrs, 4)

    def forward(self, image):
        image_features = self.image_branch(image)

        return image_features

def preprocess_image(frame):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),  # Resize to match the input size expected by the model
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
    ])
    return transform(frame).unsqueeze(0)  # Add a batch dimension

def load_models(model_paths):
    models = []
    for path in model_paths:
        model = MultimodalModel(num_landmarks=0, num_classes=4)
        model_checkpoint = torch.load(path, map_location=torch.device('cpu'))
        if isinstance(model_checkpoint, torch.nn.parallel.DataParallel):
            model_state_dict = model_checkpoint.module.state_dict()
        else:
            model_state_dict = model_checkpoint
        model.load_state_dict(model_state_dict)
        model.to(torch.device('cpu'))
        model.eval()
        models.append(model)
    return models

model_paths = ['model_mv1.pth', 'model_mv2.pth', 'model_mv3.pth', 'model_mv4.pth', 'model_mv5.pth', 'model_mv6.pth']
#model_paths = ['model_mv1.pth', 'model_mv2.pth', 'model_mv6.pth']
models = load_models(model_paths)

def predict(model, input_tensor):
    with torch.no_grad():
        outputs = model(input_tensor)
        _, preds = torch.max(outputs, 1)
    return preds.cpu().numpy()


if __name__ == '__main__':
    # giving the webpage a title
    #st.sidebar.button("Home")
    st.title("Engagement Level Classifier")
    select_pages = st.sidebar.selectbox(
        'Pages',
        ('Home', 'Engagement Classifier')
    )

    if select_pages == "Home":
        st.subheader("About This Project")
        st.markdown('<p style="color:Black;">The Myers-Briggs Type Indicator (MBTI) is a personality model developed by Katharine Cooks Briggs and Isabel Briggs Myers in 1940. The MBTI personality displays a combination of preferences from four different domains. As many as 80 percent of the top 500 companies and 89 percent of the top 100 companies in the United States use the MBTI personality test in the recruitment process. Generally, test takers need to answer about 50 to 70 questions and it is relatively expensive to know MBTI personality. To solve this problem, the researcher developed a personality classification system using the Convolutional Neural Network (CNN) methods and GloVe (Global Vectors for Word Representation) word embedding.</p>', unsafe_allow_html = True)
        st.text("")
        st.markdown('<p style="color:Black;">This project is intended for undergraduate thesis with the title of "Myers-Briggs Type Indicator (MBTI) Personality Model Classification in English Text using Convolutional Neural Network (CNN) Method".</p>', unsafe_allow_html = True)
        st.text("")
        st.markdown('<p style="color:Black;">Student Name: Joseph Ananda Sugihdharma</p>', unsafe_allow_html=True)
        st.markdown('<p style="color:Black;">Supervisor 1 Name: Dr.Eng. Fitra Abdurrachman Bachtiar, S.T., M.Eng.</p>', unsafe_allow_html=True)
        st.markdown('<p style="color:Black;">Supervisor 2 Name: Dr. Eng. Novanto Yudistira, S.Kom., M.Sc.</p>', unsafe_allow_html=True)
        st.markdown('<p style="color:Black;"><b>Last Updated: 20-Jul-2024</b></p>', unsafe_allow_html=True)

    if select_pages == "Engagement Classifier":
        # the following lines create text boxes in which the user can enter
        # the data required to make the prediction
        st.subheader("Upload Image")
        instruction = "Please upload a single image below"

        st.markdown(f'<p style="color:Black;"><b>{instruction}</b></p>',
                    unsafe_allow_html=True)

        file = st.file_uploader("Upload file", type=["csv", "png", "jpg"])
        show_file = st.empty()
        if not file:
            show_file.info("Please upload a file : {} ".format(' '.join(["csv", "png", "jpg"])))

        classifier_1 = 0
        classifier_2 = 0
        classifier_3 = 0
        classifier_4 = 0
        classifier_5 = 0
        classifier_6 = 0

        if st.button("Predict"):
            start_time = time.time()
            # Check the file type and process accordingly
            if file.type in ["image/png", "image/jpeg"]:
                image = Image.open(file)
                show_file.image(image, caption="Uploaded Image", use_column_width=True)

                # Convert image to NumPy array and preprocess
                image_array = np.array(image)
                #if image_array.ndim == 2:
                #    image_array = np.stack((image_array,) * 3, axis=-1)  # Convert grayscale to RGB
                #elif image_array.ndim == 3 and image_array.shape[2] == 1:
                #    image_array = np.concatenate([image_array] * 3, axis=-1)  # Convert single channel to RGB

                input_tensor = preprocess_image(image_array)

                y_pred_lists = []
                for model in models:
                    y_pred = predict(model, input_tensor)
                    y_pred_lists.append(y_pred)

                # Aggregate predictions using majority voting
                print("Models Predict Probabilities", y_pred_lists)
                final_predictions = []
                for i in range(len(y_pred_lists[0])):
                    votes = [preds[i] for preds in y_pred_lists]
                    majority_vote = max(set(votes), key=votes.count)
                    final_predictions.append(majority_vote)

                st.markdown(
                    f'<p style="color:Black;"><b>Final Predictions: {final_predictions}',
                    unsafe_allow_html=True)

                end_time = time.time()
                st.markdown(f'<p style="color:Black">Elapsed Time: <b>{end_time-start_time:.3f} Seconds</b></p>',
                                unsafe_allow_html=True)