import streamlit as st
import requests
import base64
import json
import numpy as np
import pandas as pd
from PIL import Image

stTopTitle=st.empty()
stTopUrl=st.sidebar.empty()
stFileChooser=st.sidebar.empty()
stImageDisplay=st.empty()
stJsonDisplay=st.empty()
stAIDisplay=st.empty()


def generate_column_names(resolution):
    column_names = []
    for a in range(1,resolution+1):
        for b in range(1,resolution+1):
            column_names.append(str(a)+'x' + str(b))
    return column_names
 
def convert_json(final_image):
  #print("\n\nStart convert_json\n",final_image)
  new_df = pd.DataFrame(data=final_image,columns=generate_column_names(28))
  pd.set_option("display.max_rows", None, "display.max_columns", None)
  #print("\n\nMID convert_json\n",new_df)
  json_file = new_df.to_dict('records')
  myJson=json_file[0]
  #print("\n\nMID2 convert_json\n",myJson)
  for k in myJson:
    v=myJson[k]
    if(v<0):
      myJson[k]=v+256
  #print("\n\nEND convert_json\n",myJson)
  return myJson

def convert_grayscale(im):
  #print("\n\nStart convert_grayscale\n",im)
  # Convert to grayscale if its a color image
  if len(im.shape) > 2 and im.shape[2]>2:
    red = im[:,:,0]
    green = im[:,:,1]
    blue = im[:,:,2]
    # Convert color to grayscale
    grayscale_image = (red * 0.299) + (green * 0.587) + (blue * 0.114)
    #print("Image was converted from RGB")
  elif len(im.shape) == 2:
    grayscale_image = im
    #print("Image was kept as-is")
  #print("\n\nEND convert_grayscale",grayscale_image,"\n")
  return grayscale_image

# This is a helper function to flatten image into a single row after downsampling the image to 28x28
def flatten_784(grayscale_image):
  #print("\n\nStart flatten\n",grayscale_image)
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
  img=Image.fromarray(downsampled_image)
  #st_text("Post-processing image")
  #st_image(img)
  new_row = list(downsampled_image.reshape(1,784))
  #print("\n\nEnd flatten\n",new_row)
  return new_row

def get_prediction_data(data,url):
  r = requests.post(url, data=json.dumps(data))
  #print("Data AI input:",data)
  response = getattr(r,'_content').decode("utf-8")
  #print("Data AI predicts:",response)
  return response

def processFile(f,url,caption):
  print("Processing uploaded file with URL:",url)
  #bytesData=f.getvalue()
  #st_text("Initial image")
  stImageDisplay.image(f,caption=caption)
  print(f)
  image=Image.open(f)
  img_array=np.array(image)
  grayscale_image=convert_grayscale(img_array)
  final_image=flatten_784(grayscale_image)
  json_data=convert_json(final_image)
  #print("JSON data",json_data)
  #st_title("Checking")
  prediction=get_prediction_data(json_data,url)
  print("\n\nData prediction",prediction)
  stJsonDisplay.json(prediction)
  predicted_label = json.loads(json.loads(prediction)['body'])['predicted_label']
  print("\n\nPredicted label", predicted_label)
  stAIDisplay.title("AI says:"+str(predicted_label))

#
# Main code
#
np.set_printoptions(linewidth=200)
version=" v 2.0"
imlist=[]
urlDefault = 'https://askai.aiclub.world/bc1fe184-efe3-4683-81f4-ededffb6c287'
stTopTitle.title("Image AI for Gateway"+version)
url=stTopUrl.text_input("URL",urlDefault)
uploadedFile=stFileChooser.file_uploader("Choose file")
if uploadedFile is not None:
  processFile(uploadedFile,url,"File upload")
  uploadedFile=None
imagelist=[ "img/112.png", "img/90.png", "img/76.png", "img/110.png", "img/84.png", "img/61.png" ]
with st.sidebar:
  for idx,x in enumerate(imagelist):
    msg=f"Image {idx+1}"
    if st.button(msg):
      processFile(imagelist[idx],url,msg)
