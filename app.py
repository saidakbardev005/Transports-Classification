import streamlit as st  
from fastai.vision.all import *
import plotly.express as px
import pathlib
import platform

plt= platform.system()
if plt == "Windows":pathlib.WindowsPath = pathlib.PosixPath
    
# title
st.title("Transportni klassifikatsiya qiluvchi model")

# rasmni joylash
file=st.file_uploader("Rasm yuklash", type=["png","jpeg","gif","svg"])

#PIL convert
img=PILImage.create(file)
if file is not None: 
    st.image(file)
    #PIL convert
    img=PILImage.create(file)
    #model
    model=load_learner("Transport_model.pkl")

    #prediction
    pred,pred_id,probs=model.predict(img)
    st.success(f"Bashorat: {pred}") 
    st.info(f"Ehtimollik: {probs[pred_id]*100:.1f}%")

    # plotting
    fig=px.bar(x=probs*100, y=model.dls.vocab)
    st.plotly_chart(fig)
