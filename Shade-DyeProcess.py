#!/usr/bin/env python
# coding: utf-8

# ## Packages
# 

# In[1]:


pip install colormath


# In[2]:


import seaborn as sns
import keras
import PIL
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors

import sklearn
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.linear_model import LinearRegression, RANSACRegressor

from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

import tensorflow
from tensorflow import keras
from keras import layers, models, activations
from keras.models import Sequential,Model
from keras.layers import Input, Dense, Activation

import colormath
from colormath.color_objects import LabColor, sRGBColor
from colormath.color_conversions import convert_color
from colormath import color_diff_matrix

#from colormath.color_diff import delta_e_cie2000

from PIL import Image
import math
import mlxtend
from mlxtend.plotting import heatmap

import warnings
warnings.filterwarnings("ignore")


# In[4]:


df = pd.read_excel('Shade-DyeProcess.xlsx',header=0)
df
#


# LAB to RGB
# 

# In[5]:


from colormath.color_objects import LabColor, sRGBColor
from colormath.color_conversions import convert_color

def lab_to_rgb(l, a, b):
    lab_color = LabColor(lab_l=l, lab_a=a, lab_b=b)
    rgb_color = convert_color(lab_color, sRGBColor)
    r = float(rgb_color.rgb_r * 255)
    g = float(rgb_color.rgb_g * 255)
    b = float(rgb_color.rgb_b * 255)
    return r, g, b

def lab_to_hex(Li, Ai, Bi):
    lab_color = LabColor(Li, Ai, Bi)
    rgb_color = convert_color(lab_color, sRGBColor)
    hex_code = rgb_color.get_rgb_hex()
    return hex_code


# In[6]:


import math

def create_abs_coeff(total_thickness, R, G, B):

    if R == 0 or G == 0 or B == 0:
        return 0

    else:
        R_1 = R / 255
        G_1 = G / 255
        B_1 = B / 255

        R_tr = float(1 - R_1)
        Max_R = float(R_tr / R_1)
        Max_G = Max_R * G_1
        Max_B = Max_R * B_1

        Transmittance = (0.2125 * R_tr) + (0.7154 * Max_G) + (0.0721 * Max_B)

        if Transmittance <= 0:
            Transmittance = -(Transmittance)

        Absorbance = 2 - math.log10(Transmittance * 100)

        Abs_coeff = (2.303 * Absorbance) / total_thickness
        
        return Abs_coeff


# In[7]:


df = pd.read_excel("Shade Prediction - TTTg.xlsx")
df["R"], df["G"], df["B"] = zip(*df.apply(lambda row: lab_to_rgb(row["Li"], row["Ai"], row["Bi"]), axis=1))
df['Hex_code'] = df.apply(lambda row: lab_to_hex(row['Li'], row['Ai'], row['Bi']), axis=1)


# In[8]:


df


# In[9]:


df["Abs_coeff"] = df.apply(lambda row: create_abs_coeff(row["Thickness"], row["R"], row["G"], row["B"]), axis=1)
df.to_excel("data_with_rgb&tg.xlsx", index=False)


# ## Load the dataset

# In[10]:


df


# In[11]:


df.isnull().sum()


# ## Data Exploration

# # 1

# In[12]:


df.describe()


# In[13]:


df.dtypes


# In[14]:


Text_df1 = df.dtypes[df.dtypes == "object"].index
df[Text_df1].describe()


# In[15]:


new_dm = pd.Categorical(df["DyeingMethod"].astype(str))
new_dm .describe()


# In[16]:


df['Duration'].unique()


# In[17]:


categorical_columns1 = df.select_dtypes(include=['object'])

unique_values_df1 = pd.DataFrame(columns=['Column', 'Unique_Values'])

for column in categorical_columns1.columns:
    unique_values1 = df[column].unique()
    unique_values_df1 = pd.concat([unique_values_df1, pd.DataFrame({'Column': [column], 'Unique_Values': [unique_values1]})], ignore_index=True)

unique_values_df1


# In[18]:


def duplicate(data):
    dup_rows = data[data.duplicated()]
    if dup_rows.empty:
        print("There are no duplicte rows in the dataset")
    else:
        print(f"There are {len(dup_rows)} in the dataset")
        data.drop_duplicates(inplace=True)
    #print(data)
duplicate(df)


# ## Data Cleaning and Preprocessing

# In[19]:


df.columns


# In[20]:


numfeatures = ['Li', 'Ai', 'Bi', 'pH', 'Temp', 'WaterBathRatio', 'Duration', 'Concentration', 'Lf', 'Af', 'Bf','Abs_coeff']

plt.figure(figsize=(15, 10))
for feature in numfeatures:
    plt.subplot(3, 4, numfeatures.index(feature) + 1)
    sns.boxplot(x=df[feature])
    plt.title("Box Plot of " + feature)
plt.tight_layout()
plt.show()


# In[21]:


df.isnull().sum()


# In[22]:


#numfeatures = ['Li', 'Ai', 'Bi', 'pH', 'Temp', 'BathRatio', 'Duration', 'Concentration', 'Lf', 'Af', 'Bf']

# Calculate skewness for each numerical column
skewness = df[numfeatures].skew()

# Plot histograms for each numerical column
fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(18, 12))
for i, column in enumerate(numfeatures):
    ax = axes[i // 4, i % 4]
    df[column].hist(ax=ax, bins=15)
    ax.set_title(f"{column} Skewness: {skewness[column]:.2f}")
    ax.set_xlabel(column)
    ax.set_ylabel("Frequency")

plt.tight_layout()
plt.show()


# In[23]:


col_mean = ['Li','Bi', 'Temp','Duration','Lf','Bf']
col_median = [ 'pH','Ai', 'WaterBathRatio',  'Concentration',  'Af','Abs_coeff']

# Fill missing values with column-wise mean
df[col_mean] = df[col_mean].fillna(df[col_mean].mean())

# Fill missing values with column-wise median
df[col_median] = df[col_median].fillna(df[col_median].median())


# In[24]:


df.columns


# In[25]:


col_mode = ['thread_group','Thread','DyeingMethod']
#col_mode = ['thread_group', 'Substrate ','Thread','DyeingMethod']

df[col_mode] = df[col_mode].fillna(df[col_mode].mode())


# In[26]:


df.isnull().sum()


# In[27]:


df.drop(df[(df['Li'] > 100)  & (df['Li'] < 0)].index, inplace=True)
df.drop(df[(df['Lf'] > 100) & (df['Li'] < 0)].index, inplace=True)


# In[28]:


#df.to_excel("data_with_rgb.xlsx", index=False)


# In[29]:


df


# ## Feature Extraction

# In[30]:


# Function to convert LAB values to Hex code
#Lab -> RGB -> hexCode

def hex_to_rgb(hex_code):
    hex_code = hex_code.lstrip('#')
    return tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))

#Hexcode to RGB, image using RGB
def create_color_image(hex_code, width=100, height=100):
    rgb = hex_to_rgb(hex_code)
    image = Image.new('RGB', (width, height), rgb)
    return image

def ttype(thread):
    if thread == 'Single Fibre' or 'singlefibre':
        return 0.025
    elif thread == 'Coarse' or 'coarse':
        return 0.127


# In[31]:


#https://python-colormath.readthedocs.io/en/latest/_modules/colormath/color_diff.html#delta_e_cie1976
def _get_lab_color1(color):
    if not color.__class__.__name__ == 'LabColor':
        raise ValueError("Delta E functions can only be used with two LabColor objects.")
    return np.array([color.lab_l, color.lab_a, color.lab_b])

def _get_lab_color2(color):
    if not color.__class__.__name__ == 'LabColor':
        raise ValueError("Delta E functions can only be used with two LabColor objects.")
    return np.array([(color.lab_l, color.lab_a, color.lab_b)])

def delta_e_cie1976(color1, color2):
    color1 = _get_lab_color1(color1)
    color2 = _get_lab_color2(color2)
    delta_e = color_diff_matrix.delta_e_cie1976(color1, color2)
    return np.float64(delta_e)

def color(L,A,B):
    color = LabColor(lab_l=L, lab_a=A, lab_b=B)
    return color
#color2 = LabColor(lab_l=L, lab_a=A, lab_b=B)


# In[32]:


df.columns


# ## Data

# In[33]:


df.columns


# In[34]:


from sklearn.preprocessing import LabelEncoder

# Create LabelEncoder instances for 'DyeingMethod' and 'Thread'
DyeingMethod_encoder = LabelEncoder()
Thread_encoder = LabelEncoder()
thread_group_encoder = LabelEncoder()
#Substrate_encoder = LabelEncoder()

# Encode the 'DyeingMethod' and 'Thread' columns in place
df['DyeingMethod'] = DyeingMethod_encoder.fit_transform(df['DyeingMethod'])
df['Thread'] = Thread_encoder.fit_transform(df['Thread'])
df['thread_group'] = thread_group_encoder.fit_transform(df['thread_group'])
#df['Substrate '] = Substrate_encoder.fit_transform(df["Substrate "])

# Define a list of columns and their corresponding encoders for printing
columns_and_encoders = [
    ('DyeingMethod', DyeingMethod_encoder),
    ('Thread', Thread_encoder),
    ('thread_group',thread_group_encoder),
    #('Substrate ',Substrate_encoder)
]

# Iterate through the columns and encoders and print the mapping
for column, encoder in columns_and_encoders:
    encoded_values = df[column].unique()
    print(f"Column: {column}")
    for encoded in encoded_values:
        decoded_value = encoder.inverse_transform([encoded])[0]
        print(f"Encoded: {encoded}, Decoded: {decoded_value}")
    print("\n")


# In[35]:


df.columns


# ## RANSAC

# In[36]:


X = df[['Li', 'Ai', 'Bi', 'Concentration', 'pH', 'Temp', 'WaterBathRatio', 'DyeingMethod', 'Duration', 'Thread','Thickness','thread_group','Abs_coeff']]
Y = df[['Lf', 'Af', 'Bf']]


# In[37]:


#X = df[['Li', 'Ai', 'Bi', 'Concentration', 'pH', 'Temp', 'WaterBathRatio', 'DyeingMethod', 'Duration', 'Substrate ','Thread','Thickness','thread_group','Abs_coeff']]
#Y = df[['Lf', 'Af', 'Bf']]


# In[38]:


X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.3, random_state=42)

# RANSAC Regression
ransac = RANSACRegressor(min_samples=100, max_trials=80,loss='squared_error', random_state=40,residual_threshold=10)
ransac.fit(X_train, Y_train)
ransac_pred = ransac.predict(X_test)
ransac_mse = mean_squared_error(Y_test, ransac_pred)
print("RANSAC Regression MSE:", ransac_mse)


# In[39]:


# Calculate the R-squared score to measure accuracy
r2 = r2_score(Y_test, ransac_pred)
print("R-squared:", r2)


# In[41]:


I_Shade = input("Enter Initial Shade:\t")
Li, Ai, Bi = I_Shade.split(",")[:3]
R,G,B = lab_to_rgb(Li, Ai, Bi)
Concentration = input("Enter concentration of the shade:\t")
pH = input("Enter pH:\t")
Temp = input("Enter Temperature:\t")
WaterBathRatio = input("Enter WaterBathRatio:\t")
Duration = input("Enter Duration:\t")
DyeingMethod = input("Enter DyeingMethod:\t")
#Substrate = input("Enter the Substrate:\t")
Thread = input("Enter the type of thread:\t")
Thickness = ttype(Thread)
thread_group = input("Enter the thread group:\t")
Abs_coeff = create_abs_coeff(Thickness, R,G,B)
ran_prediction = ransac.predict([[float(Li),float(Ai), float(Bi), float(Concentration), float(pH), int(Temp), float(WaterBathRatio), int(DyeingMethod_encoder.transform([DyeingMethod])),int(Duration),
                             int(Thread_encoder.transform([Thread])), float(Thickness), int(thread_group_encoder.transform([thread_group])),float(Abs_coeff)]])

#ran_prediction = ransac.predict([[float(Li),float(Ai), float(Bi), float(Concentration), float(pH), int(Temp), float(WaterBathRatio), int(DyeingMethod_encoder.transform([DyeingMethod])),int(Duration),
#                             int(Substrate_encoder.transform([Substrate])),int(Thread_encoder.transform([Thread])), float(Thickness), int(thread_group_encoder.transform([thread_group])),float(Abs_coeff)]])

L, A, B = ran_prediction[0]
color2 = color(L, A, B)
color1 = color(Li, Ai, Bi)
print(f'L: {L}, A: {A}, B: {B}\n')
hex_code = lab_to_hex(L, A, B)
image = create_color_image(hex_code)
print("Predicted Shade")
image


# In[42]:


delta_e = delta_e_cie1976(color1, color2)
print(f"Color Difference {delta_e}")

hex_code = lab_to_hex(Li, Ai, Bi)
image = create_color_image(hex_code)
image


# In[43]:


import joblib


# In[44]:


from joblib import dump


# In[46]:


dump(ransac,'C:/Users/HP/djlab/savedModels/RANSAC1.joblib')


# In[ ]:




