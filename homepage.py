import streamlit as st
from PIL import Image
import numpy as np
import json

st.title('Interpretability of Saliency-Based Models')

st.write("""The goal of this project is to explore the interpretability of saliency-based models.
        The first step is to choose some basic options related to the explainability model you will be using
        This includes choosing the saliency method, the model and the image you want to explain. This is done
        on the page titled 'Initialise the Demo - I'. Therafter, you can customise the demo further in the
        second part of the demo on the page titled 'Initialise the Demo - II' where you can choose options
        like number of steps, baseline, etc. Finally, you can run the demo on the page titled 'Run the Demo'.
        You can also directly run the demo by clicking on the 'Run the Demo' button on the sidebar, in which case
        a default selection of parameters will be chosen""")

with open('savestate.json', 'a+', encoding='utf-8') as f:
    f.seek(0)
    data = json.load(f)
    if (data.get("run", "No") == "No"):
        with open('savestate.json', 'w', encoding='utf-8') as g:
            data = {
                "run": "Yes",
                "model": "ResNet50V2",
                "method": "Integrated Gradients",
                "smoothgrad": "No",
                "idgi": "No",
                "image": "Default",
                "class": "Top Class",
                "classnum": -1,
                "steps": 20,
                "baseline": "black",
                "max_sig": 0.5,
                "grad_step": 0.01,
                "sqrt": "No",
                "noise_steps": 20,
                "noise_var": 0.1,
                "steps_at": "-",
            }
            json.dump(data, g, ensure_ascii=False, indent=4)

def init_demo():
    st.switch_page("pages/Initialise the Demo - I.py")
def run_demo():
    st.switch_page("pages/Run the Demo.py")

if st.button("Initialise the Demo - I"):
    init_demo()
if st.button("Run the Demo"):
    run_demo()
