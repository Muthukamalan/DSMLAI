

#Importing all the necessary libraries
import torch
import requests
import numpy as np
import pandas as pd
import gradio as gr
from io import BytesIO
from PIL import Image as PILIMAGE

from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
from sentence_transformers import SentenceTransformer, util



device = "cuda" if torch.cuda.is_available() else "cpu"

# Define model
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

# Load data
photos = pd.read_csv("./photos.tsv000", sep='\t', header=0)
photo_features = np.load("./features.npy")
photo_ids = pd.read_csv("./photo_ids.csv")
photo_ids = list(photo_ids['photo_id'])



def encode_text(text):
  with torch.no_grad():
    # Encode and normalize the description using CLIP
    inputs = tokenizer([text],  padding=True, return_tensors="pt")
    inputs = processor(text=[text], images=None, return_tensors="pt", padding=True).to(device=device)
  text_encoded =  model.get_text_features(**inputs).detach().cpu().numpy()
  return text_encoded

def encode_image(image):
  image = PILIMAGE.fromarray(image.astype('uint8'), 'RGB')
  with torch.no_grad():
        photo_preprocessed = processor(text=None, images=image, return_tensors="pt", padding=True)["pixel_values"]
        search_photo_feature = model.get_image_features(photo_preprocessed.to(device))
        search_photo_feature /= search_photo_feature.norm(dim=-1, keepdim=True)
  image_encoded = search_photo_feature.cpu().numpy()
  return image_encoded

T2I = "Text2Image"
I2I = "Image2Image"

def similarity(feature, photo_features):
  similarities = list((feature @ photo_features.T).squeeze(0))
  return similarities

def find_best_matches(image, mode, text):
  # Compute the similarity between the description and each photo using the Cosine similarity
  print ("Mode now ",mode) 

  if mode == "Text2Image":
    # Encode the text input
    text_features = encode_text(text)
    feature = text_features
    similarities = similarity(text_features, photo_features)
    
    
  else:
    #Encode the image input
    image_features = encode_image(image)
    feature = image_features
    similarities = similarity(image_features, photo_features)
    
  # Sort the photos by their similarity score
  best_photos = sorted(zip(similarities, range(photo_features.shape[0])), key=lambda x: x[0], reverse=True)
  
  matched_images = []
  for i in range(3):
    # Retrieve the photo ID
    idx = best_photos[i][1]
    photo_id = photo_ids[idx]
    
    # Get all metadata for this photo
    photo_data = photos[photos["photo_id"] == photo_id].iloc[0]
    
    # Display the images
    #display(Image(url=photo_data["photo_image_url"] + "?w=640"))
    response = requests.get(photo_data["photo_image_url"] + "?w=640")
    img = PILIMAGE.open(BytesIO(response.content))
    matched_images.append(img)
  return matched_images




demo = gr.Interface(
    fn=find_best_matches,
    inputs=[
        gr.Image(label="Image to search",),# optional=True
        gr.Radio([T2I, I2I]),
        gr.Textbox(lines=1, label="Text query", placeholder="Introduce the search text...",)
      ],
      theme="grass",
      outputs=[
        gr.Gallery(label="Generated images", show_label=False, elem_id="gallery")
      ],
      title="CLIP Search",
      description="This application displays TOP THREE images from Unsplash dataset that best match the search query provided by the user from (25k images-db). Moreover, the input can be provided via two modes ie text or image form.",
      examples=[
        ["./assets/duck.jpeg","Image2Image", None] ,
        [None, "Text2Image", "Planet Earth"],
        ["./assets/horse.jpeg", "Text2Image", "Horse"]

      ]
    )


with open("README.md", "r+") as file: 
  readme_content = file.read()
# 🏐⚽🏀🎾🤸
readme =gr.Interface( fn = None, inputs=None, outputs=gr.Markdown(readme_content[150:]),clear_btn=None, css="footer{display:none !important}",flagging_options=[],show_progress='hidden',title="") #gr.Interface(lambda name: "Bye " + name, "text", "text")#


app = gr.TabbedInterface([demo, readme ],tab_names=["CoolCLIP 🦆","README"])
app.launch(debug=False,)
