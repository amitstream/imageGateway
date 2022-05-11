import streamlit as st
import requests
import base64
import json
import numpy as np
from PIL import Image

def get_prediction_img(image_data):
  url = 'https://askai.aiclub.world/39a4f3a3-e637-4981-a88c-b2597ab12be0'
  r = requests.post(url, data=image_data)
  response = r.json()['predicted_label']
  print("Image AI predicts:",response)
  return response

def get_prediction_data(data,url):
#  url = 'https://d3yowc8vr7.execute-api.us-east-1.amazonaws.com/Predict/13d5ab46-b369-4c84-966e-41a0c3ed83d1'
#  url = 'https://askai.aiclub.world/bc1fe184-efe3-4683-81f4-ededffb6c287'
  r = requests.post(url, data=json.dumps(data))
  response = getattr(r,'_content').decode("utf-8")
  print("Data AI predicts:",response)
  return response

def processFile(f,url):
  print("Got file upload")
  bytesData=f.getvalue()
  st.image(f)
  image=Image.open(f)
  img_array=np.array(image)
  grayscale_image=convert_grayscale(img_array)
  final_image=flatten_784(grayscale_image)
  print("Final image",final_image)
  #st.title("Checking")
  prediction=get_prediction_data(final_image,url)
  print("\n\nData prediction",prediction)
  predicted_label = json.loads(json.loads(prediction)['body'])['predicted_label']
  print("\n\nPredicted label", predicted_label)
  st.title("AI says:"+str(predicted_label))

  #payload = base64.b64encode(bytesData)
  #response = get_prediction_img(payload)
  #print("\n\nResponse is:",response)
  #st.title("IMAGE AI says:"+response)

def convert_grayscale(im):
  # Convert to grayscale if its a color image
  if len(im.shape) > 2 and im.shape[2]>2:
    red = im[:,:,0]
    green = im[:,:,1]
    blue = im[:,:,2]
    # Convert color to grayscale
    grayscale_image = (red * 0.299) + (green * 0.587) + (blue * 0.114)
  elif len(im.shape) == 2:
    grayscale_image = im
  return grayscale_image

# This is a helper function to flatten image into a single row after downsampling the image to 28x28
def flatten_784(grayscale_image):
  # Find the width and length of the image
  num_rows_image = grayscale_image.shape[0]
  num_cols_image = grayscale_image.shape[1]
  # Figure out the downsampling value for each dimension
  downsample_rows = int(np.floor(num_rows_image/28))
  downsample_cols = int(np.floor(num_cols_image/28))

  # Downsample it
  downsampled_image = grayscale_image[::downsample_rows,::downsample_cols]
  # Somtimes, the dimensions after downsampling are not accurate, pick the first 28 pixels in each direction
  downsampled_image = downsampled_image[0:28,0:28]
  # Convert the vector to a list
  list_image = list(downsampled_image.reshape(784,))
  #From the list, create a dictionary
  e=0
  d={}
  for i in range(1,29):
    for j in range(1,29):
      l=f"{i}x{j}"
      d[l]=e
      e=e+1
  return d

urlDefault = 'https://askai.aiclub.world/bc1fe184-efe3-4683-81f4-ededffb6c287'
st.title("Image AI for Gateway")
url=st.text_input("URL",urlDefault)
uploadedFile=st.file_uploader("Choose file")
if uploadedFile is not None:
  processFile(uploadedFile,url)
