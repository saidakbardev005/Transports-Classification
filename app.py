import streamlit as st  
from fastai.vision.all import *
import plotly.express as px
import pathlib
import platform
import io
#temp=pathlib.PosixPath
#pathlib.PosixPath=pathlib.WindowsPath
plt=platform.system()
if plt =="Linux" or plt == "Darwin":
  pathlib.WindowsPath=pathlib.PosixPath
# title
st.title("Transportni klassifikatsiya qiluvchi model")

# rasmni joylash
file=st.file_uploader("Rasm yuklash", type=["png","jpeg","gif","svg"])

#PIL convert
img=PILImage.create(file)
try:
  
  if file: 
    st.image(file)
    #model
    model=load_learner("Transport_model.pkl")

    #prediction
    pred,pred_id,probs=model.predict(img)
    st.success(f"Bashorat: {pred}") 
    st.info(f"Ehtimollik: {probs[pred_id]*100:.1f}%")

    # plotting
    fig=px.bar(x=probs*100, y=model.dls.vocab)
    st.plotly_chart(fig)
  else:
    st.write("Iltimos, rasm yuklang.")
except Exception as e:
    st.error(f"Xatolik yuz berdi: {e}")
