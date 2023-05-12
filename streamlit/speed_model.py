import streamlit as st

import src.config
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 

# modeling packages
import statsmodels as sm
from sklearn import preprocessing, pipeline, compose
from sklearn import linear_model, model_selection, svm
from sklearn import metrics


st.markdown("# Exploratory Data Analysis")
st.sidebar.markdown("# Exploratory Data Analysis")
st.write("""
Our goal for conducting EDA was to identify new features
that could be use to improve upon the provided Linear Model.
Particularly, we were interested in seeing whether other features
including, time, network conditions, testing frequency among others
could be included in the regression analysis.
""")

st.markdown("## Distribution Analyses")
st.sidebar.markdown("## Distribution Analyses️")

data_name = "BestEstimate_On_DissolvedSmallerCitiesHexes_Time"
data_dir = src.config.DATA_DIRECTORY / "processed" / "statistical_geometries" / "time"
df = pd.read_csv(data_dir / (data_name+".csv"))
st.dataframe(dataframe)

st.write("""

""")

st.markdown("## Correlation Analyses")
st.sidebar.markdown("## Correlation Analyses️")


st.markdown("# Improving the Simple Linear Model")
st.sidebar.markdown("# Improving the Simple Linear Model")




st.write("# Hello!")


