#!/usr/bin/env python
# coding: utf-8

# ## Packages
# 

# In[1]:


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
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.linear_model import LinearRegression, RANSACRegressor
from sklearn.neighbors import KNeighborsRegressor

import colormath
from colormath.color_objects import LabColor, sRGBColor
from colormath.color_conversions import convert_color
from colormath import color_diff_matrix

#from colormath.color_diff import delta_e_cie2000

from PIL import Image

import mlxtend
from mlxtend.plotting import heatmap
import colorspacious
import colorsys

import warnings
warnings.filterwarnings("ignore")


# ## Load the dataset

# In[2]:


#df = pd.read_excel('Shade Prediction - CS.xlsx',header=0)
df = pd.read_excel('Quality Dye2 - 2 - 3 - Copy.xlsx',header=0)
df


# ## Data Exploration

# # 1

# In[3]:


df.describe()


# In[4]:


df.dtypes


# In[5]:


Text_df1 = df.dtypes[df.dtypes == "object"].index
df[Text_df1].describe()


# In[6]:


new_dm = pd.Categorical(df["Fastness_Type"].astype(str))
new_dm .describe()


# In[3]:


new_dm = pd.Categorical(df["Chemical"].astype(str))
new_dm .describe()


# In[4]:


df['D_Duration'].unique()


# In[5]:


df.columns


# In[6]:


counts = df['Computer Strength'].value_counts()
counts


# In[7]:


df['Temp'].unique()


# In[8]:


df['pH'].unique()


# In[9]:


df['Chemical_Conc'].unique()


# In[10]:


categorical_columns1 = df.select_dtypes(include=['object'])

unique_values_df1 = pd.DataFrame(columns=['Column', 'Unique_Values'])

for column in categorical_columns1.columns:
    unique_values1 = df[column].unique()
    unique_values_df1 = pd.concat([unique_values_df1, pd.DataFrame({'Column': [column], 'Unique_Values': [unique_values1]})], ignore_index=True)

unique_values_df1


# In[11]:


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

# In[12]:


df.columns


# In[13]:


numfeatures = ['L', 'A', 'B', 'pH', 'Temp', 'Thickness', 'D_Duration', 'Washings', 'Chemical_Conc',
       'Computer Strength']

plt.figure(figsize=(15, 10))
for feature in numfeatures:
    plt.subplot(3, 4, numfeatures.index(feature) + 1)
    sns.boxplot(x=df[feature])
    plt.title("Box Plot of " + feature)
plt.tight_layout()
plt.show()


# In[14]:


df.isnull().sum()


# In[15]:


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


# In[16]:


numfeatures = ['L', 'A', 'B', 'pH', 'Temp', 'Thickness', 'D_Duration', 'Washings', 'Chemical_Conc',
       'Color Strengtth / Computer Strength']
#Index(['L', 'A', 'B', 'pH', 'Temp', 'Thickness', 'Thread', 'thread_group',
      # 'D_Duration', 'Fastness_Type', 'Washings', 'Chemical', 'Chemical_Conc',
      # 'Color Strengtth / Computer Strength'],
     # dtype='object')


# In[17]:


col_mean = ['D_Duration','Washings','Temp']
col_median = ['L', 'A', 'B', 'Chemical_Conc','pH','Thickness',   'Computer Strength']

# Fill missing values with column-wise mean
df[col_mean] = df[col_mean].fillna(df[col_mean].mean())

# Fill missing values with column-wise median
df[col_median] = df[col_median].fillna(df[col_median].median())


# In[18]:


col_mode = ['thread_group',  'Fastness_Type', 'Thread','Chemical','Lubricant ']

df[col_mode] = df[col_mode].fillna(df[col_mode].mode())


# In[19]:


df.isnull().sum()


# In[20]:


df.drop(df[(df['L'] > 100)].index, inplace=True)


# ## Feature Extraction

# In[21]:


# Function to convert LAB values to Hex code
#Lab -> RGB -> hexCode
def lab_to_hex(Li, Ai, Bi):
    lab_color = LabColor(Li, Ai, Bi)
    rgb_color = convert_color(lab_color, sRGBColor)
    hex_code = rgb_color.get_rgb_hex()
    return hex_code

def hex_to_rgb(hex_code):
    hex_code = hex_code.lstrip('#')
    return tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))

#Hexcode to RGB, image using RGB
def create_color_image(hex_code, width=100, height=100):
    rgb = hex_to_rgb(hex_code)
    image = Image.new('RGB', (width, height), rgb)
    return image


# In[22]:


#https://python-colormath.readthedocs.io/en/latest/_modules/colormath/color_diff.html#delta_e_cie1976
#Array
def _get_lab_color1(color):
    if not color.__class__.__name__ == 'LabColor':
        raise ValueError("Delta E functions can only be used with two LabColor objects.")
    return np.array([color.lab_l, color.lab_a, color.lab_b])

#Matrix
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


# In[23]:


# Apply the function to create a new 'Hex_code' column
df['Hex_code'] = df.apply(lambda row: lab_to_hex(row['L'], row['A'], row['B']), axis=1)
df


# In[24]:


df.columns


# ## Data

# In[25]:


#  'thread_group',  'Fastness_Type', 'Thread','Chemical']


# In[26]:


from sklearn.preprocessing import LabelEncoder

Chemical_encoder = LabelEncoder()
Thread_encoder = LabelEncoder()
thread_group_encoder = LabelEncoder()
Fastness_encoder = LabelEncoder()
Lubricant_encoder = LabelEncoder()

df['Chemical'] = Chemical_encoder.fit_transform(df['Chemical'])
df['Thread'] = Thread_encoder.fit_transform(df['Thread'])
df['thread_group'] = thread_group_encoder.fit_transform(df['thread_group'])
df['Fastness_Type'] = Fastness_encoder.fit_transform(df["Fastness_Type"])
df['Lubricant '] = Lubricant_encoder.fit_transform(df['Lubricant '])

columns_and_encoders = [
    ('Chemical', Chemical_encoder),
    ('Thread', Thread_encoder),
    ('thread_group',thread_group_encoder),
    ('Fastness_Type',Fastness_encoder),
    ('Lubricant ',Lubricant_encoder)
]


for column, encoder in columns_and_encoders:
    encoded_values = df[column].unique()
    print(f"Column: {column}")
    for encoded in encoded_values:
        decoded_value = encoder.inverse_transform([encoded])[0]
        print(f"Encoded: {encoded}, Decoded: {decoded_value}")
    print("\n")


# In[27]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#Index(['L', 'A', 'B', 'pH', 'Temp', 'Thickness', 'Thread', 'thread_group',
      # 'D_Duration', 'Fastness_Type', 'Washings', 'Chemical', 'Chemical_Conc',
      # 'Color Strengtth / Computer Strength'],
     # dtype='object')
cols = ['L', 'A', 'B', 'pH', 'Temp', 'Thickness', 'Thread', 'thread_group',
      'D_Duration', 'Fastness_Type', 'Washings', 'Chemical', 'Chemical_Conc',
       'Computer Strength', 'Lubricant ']

cm = np.corrcoef(df[cols].values.T)

plt.figure(figsize=(10, 11))
sns.heatmap(cm, cbar=True, annot=True, square=True, fmt=".2f", annot_kws={"size": 10},yticklabels=cols, xticklabels=cols, cmap="coolwarm")
plt.title('Correlation Heatmap')
plt.show()


# In[32]:


#X = df[['Li', 'Ai', 'Bi', 'Concentration', 'pH', 'Temp', 'WaterBathRatio', 'DyeingMethod', 'Duration']]
#Y = df[['Lf', 'Af', 'Bf']]


# # DT

# In[60]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

X = df[['L', 'A', 'B', 'pH', 'Temp', 'Thickness', 'Thread', 'thread_group',
        'D_Duration', 'Fastness_Type', 'Washings', 'Chemical', 'Chemical_Conc', 'Lubricant ']]
Y = df['Computer Strength']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=30)

dtmodel = DecisionTreeRegressor()

dtmodel.fit(X_train, y_train)

y_pred_tree = dtmodel.predict(X_test)

mse_tree = mean_squared_error(y_test, y_pred_tree)

r2_tree = r2_score(y_test, y_pred_tree)

print("Decision Tree Regression,")
print("Mean Squared Error:", mse_tree)


# In[61]:


y_pred_tree = dtmodel.predict(X_test)

mse_tree = mean_squared_error(y_test, y_pred_tree)

r2_tree = r2_score(y_test, y_pred_tree)
rmse_tree = np.sqrt(mse_tree)

print('Mean squared error: ', mse_tree)
print('Root mean squared error: ', rmse_tree)
print('R2 score: ', r2_tree)


# In[38]:


L = 40
A = 75.86
B = -78.56
pH = 11 #inc 
Temp = 75
Thickness = 0.015
Thread = "Single Fibre" # 1 ponit increase
thread_group = "TRP" 
D_Duration = 25 # 1 ponit increase
Fastness_Type = "NORMAL"
Washings = 15
Chemical = "Post-Mordanting with copper sulphate"
Chemical_Conc = 15
Lubricant = 'L3'

mlr_cs = dtmodel.predict([[float(L),float(A), float(B), float(pH), int(Temp), float(Thickness), int(Thread_encoder.transform([Thread])), int(thread_group_encoder.transform([thread_group])),
                             int(D_Duration),int(Fastness_encoder.transform([Fastness_Type])), int(Washings),int(Chemical_encoder.transform([Chemical])),float(Chemical_Conc),float(Lubricant_encoder.transform([Lubricant]))]])

hex_code = lab_to_hex(L, A, B)
image = create_color_image(hex_code)
image

#print(f"Color Strength {mlr_cs_percentage}% for the above shade")


# In[ ]:





# In[62]:


print("R Square",r2_tree)


# In[36]:


# Create LabelEncoder instances for 'Chemical' and 'Thread'
'''
Chemical_encoder = LabelEncoder()
Thread_encoder = LabelEncoder()
thread_group_encoder = LabelEncoder()
Fastness_encoder = LabelEncoder()
'''
#63.5700	-31.7600	54.810	15.0	5.0	70	0.066	40	POM with CuSo4
#63.2700	-31.5000	55.490
#62.53	-9.05	-44.637	5	70	0.025	Single Fibre	CFP-COR-A	20	CORE DYED	10	Post-Mordanting with copper sulphate 	15	98	62.23	-8.79	-43.957

#DyeingMethod_encoder.transform([Dyeing_Method])[0]
#53.34	4.65	10.81	9	60	0.076	Coarse	IBN MEDIUM	20	CORE DYED	10	Directly dyed without mordant	3.5	0.95			
#66.1	-23.15	53.57	9.5	70	0.127	Coarse	CFP-MEDIUM	10	HIGH WASH FAST	15	Post-Mordanting with copper sulphate	15	1.00			
#53.34	4.65	10.81	9	60	0.076	Coarse	IBN MEDIUM	20	CORE DYED	10	Directly dyed without mordant	3.5	L3	0.95		

#tESTING -- 40	75.86	-78.56	11	75	0.015	Single Fibre	TRP	25	NORMAL	15	Post-Mordanting with copper sulphate	15	L3	1.00		


# In[43]:


#63.5700	-31.7600	54.810	15.0	5.0	70	0.066	40	POM with CuSo4
#63.2700	-31.5000	55.490
#62.53	-9.05	-44.637	5	70	0.025	Single Fibre	CFP-COR-A	20	CORE DYED	10	Post-Mordanting with copper sulphate 	15	98	62.23	-8.79	-43.957

#DyeingMethod_encoder.transform([Dyeing_Method])[0]
#53.34	4.65	10.81	9	60	0.076	Coarse	IBN MEDIUM	20	CORE DYED	10	Directly dyed without mordant	3.5	0.95			
#66.1	-23.15	53.57	9.5	70	0.127	Coarse	CFP-MEDIUM	10	HIGH WASH FAST	15	Post-Mordanting with copper sulphate	15	1.00			
#53.34	4.65	10.81	9	60	0.076	Coarse	IBN MEDIUM	20	CORE DYED	10	Directly dyed without mordant	3.5	L3	0.95		

#tESTING -- 40	75.86	-78.56	11	75	0.015	Single Fibre	TRP	25	NORMAL	15	Post-Mordanting with copper sulphate	15	L3	1.00		

L = 53.34
A = 4.65
B = 10.81
pH = 9 #inc 
Temp = 60
Thickness = 0.076
Thread = "Coarse" # 1 ponit increase
thread_group = "IBN MEDIUM" 
D_Duration = 20 # 1 ponit increase
Fastness_Type = "CORE DYED"
Washings = 10
Chemical = "Directly dyed without mordant"
Chemical_Conc = 3.5
Lubricant = 'L3'

mlr_cs = dtmodel.predict([[float(L),float(A), float(B), float(pH), int(Temp), float(Thickness), int(Thread_encoder.transform([Thread])), int(thread_group_encoder.transform([thread_group])),
                             int(D_Duration),int(Fastness_encoder.transform([Fastness_Type])), int(Washings),int(Chemical_encoder.transform([Chemical])),float(Chemical_Conc),float(Lubricant_encoder.transform([Lubricant]))]])

#print(mlr_cs)
#mlr_cs_percentage = (mlr_cs * 100)
mlr_cs_percentage = max(0, min(mlr_cs * 100, 100))

hex_code = lab_to_hex(L, A, B)
image = create_color_image(hex_code)
image

print(f"Color Strength {mlr_cs_percentage}% for the above shade")


# In[ ]:





# In[44]:


from joblib import dump


# In[45]:


dump(dtmodel,'C:/Users/HP/djlab/savedModels/DT.joblib')


# ## Linear Regression

# In[63]:


# Model initialization
regression_model = LinearRegression()
# Fit the data(train the model)
regression_model.fit(X, Y)
# Predict
y_predicted = regression_model.predict(X)

# model evaluation
mse = mean_squared_error(Y, y_predicted)
r2 = r2_score(Y, y_predicted)
rmse = np.sqrt(mse)
# printing values
print('Slope:' ,regression_model.coef_)
print('Intercept:', regression_model.intercept_)
print('Mean squared error: ', mse)
print('Root mean squared error: ', rmse)
print('R2 score: ', r2)


# In[35]:


#63.5700	-31.7600	54.810	15.0	5.0	70	0.066	40	POM with CuSo4
#63.2700	-31.5000	55.490
#62.53	-9.05	-44.637	5	70	0.025	Single Fibre	CFP-COR-A	20	CORE DYED	10	Post-Mordanting with copper sulphate 	15	98	62.23	-8.79	-43.957

#DyeingMethod_encoder.transform([Dyeing_Method])[0]
#53.34	4.65	10.81	9	60	0.076	Coarse	IBN MEDIUM	20	CORE DYED	10	Directly dyed without mordant	3.5	0.95			
#66.1	-23.15	53.57	9.5	70	0.127	Coarse	CFP-MEDIUM	10	HIGH WASH FAST	15	Post-Mordanting with copper sulphate	15	1.00			
#53.34	4.65	10.81	9	60	0.076	Coarse	IBN MEDIUM	20	CORE DYED	10	Directly dyed without mordant	3.5	L3	0.95		

#tESTING -- 40	75.86	-78.56	11	75	0.015	Single Fibre	TRP	25	NORMAL	15	Post-Mordanting with copper sulphate	15	L3	1.00		

L = 53.34
A = 4.65
B = 10.81
pH = 9 #inc 
Temp = 60
Thickness = 0.076
Thread = "Coarse" # 1 ponit increase
thread_group = "IBN MEDIUM" 
D_Duration = 20 # 1 ponit increase
Fastness_Type = "CORE DYED"
Washings = 10
Chemical = "Directly dyed without mordant"
Chemical_Conc = 3.5
Lubricant = 'L3'

mlr_cs = regression_model.predict([[float(L),float(A), float(B), float(pH), int(Temp), float(Thickness), int(Thread_encoder.transform([Thread])), int(thread_group_encoder.transform([thread_group])),
                             int(D_Duration),int(Fastness_encoder.transform([Fastness_Type])), int(Washings),int(Chemical_encoder.transform([Chemical])),float(Chemical_Conc),float(Lubricant_encoder.transform([Lubricant]))]])

#print(mlr_cs)
#mlr_cs_percentage = (mlr_cs * 100)
mlr_cs_percentage = max(0, min(mlr_cs * 100, 100))

hex_code = lab_to_hex(L, A, B)
image = create_color_image(hex_code)
image

print(f"Color Strength {mlr_cs_percentage}% for the above shade")


# ## GBRT

# In[ ]:


'''
Mean squared error:  3.816364568493697e-05
Root mean squared error:  0.006177673161064525
R2 score:  0.9165615373634776

DT --
Mean squared error:  8.333333333333348e-07
Root mean squared error:  0.0009128709291752777
R2 score:  0.9980078358456737
'''


# In[70]:


import matplotlib.pyplot as plt

# Labels for the algorithms
algorithms = ['Linear Regression', 'Decision Tree Regression']

# RMSE values
rmse_values = [rmse,rmse_tree]

# MSE values
mse_values = [mse, mse_tree]

# R^2 values
r2_values = [r2, r2_tree]

# Bar chart
width = 0.3
x = range(len(algorithms))

plt.bar(x, rmse_values, width, label='RMSE')
plt.bar([i + width for i in x], mse_values, width, label='MSE')
plt.bar([i + 2 * width for i in x], r2_values, width, label='R^2')

plt.xlabel('Algorithms')
plt.ylabel('Values')
plt.ylim(0, 10)
plt.xticks([i + width for i in x], algorithms)
plt.title('Performance Comparison of Linear Regression and Decision Tree Regression')
plt.legend()
plt.show()


# In[69]:


import matplotlib.pyplot as plt


# Labels for the algorithms
algorithms = ['Linear Regression', 'Decision Tree Regression']

# Metrics to plot
metrics = ['RMSE', 'MSE', 'R^2']

# Values for each algorithm and metric
values = {
    'Linear Regression': [rmse, mse, r2],
    'Decision Tree Regression': [rmse_tree, mse_tree, r2_tree]
}

# Line chart
x = range(len(metrics))

for algorithm in algorithms:
    algorithm_values = values[algorithm]
    plt.plot(x, algorithm_values, marker='o', label=algorithm)



plt.xlabel('Metrics')
plt.ylabel('Values')
plt.xticks(x, metrics)
plt.title('Performance Comparison of Linear Regression and Decision Tree Regression')
plt.legend()
plt.show()


# In[72]:


from tabulate import tabulate

# Define the data
data = [
    ["Linear Regression", 3.816364568493697e-05, 0.006177673161064525, 0.9165615373634776],
    ["Decision Tree Regression", 8.333333333333348e-07, 0.0009128709291752777, 0.9890078358456737]
]

# Define headers
headers = ["Algorithm", "MSE", "RMSE", "R^2"]

# Create the table
table = tabulate(data, headers, tablefmt="pretty")

# Print the table
print(table)


# In[ ]:





# In[ ]:





# ## Multiple Prediction 

# In[ ]:


times = int(input("Enter the number Predictions: "))
for _ in range(times):
    Li = float(input("Enter Li "))
    Ai = float(input("Enter Ai "))
    Bi = float(input("Enter Bi "))
    Concentration = float(input("Enter Concentration "))
    pH = float(input("Enter pH "))
    Temp = int(input("Enter Temperature "))
    BathRatio = float(input("Enter BathRatio "))
    Duration = int(input("Enter Duration "))
    Dyeing_Method = input("Enter Dyeing_Method ")
    prediction = ransac.predict([[Li,Ai, Bi, Concentration, pH, Temp, WaterBathRatio,Duration, int(DyeingMethod_encoder.transform([DyeingMethod])[0])]])
    print('Input Shade')
    hex_code = lab_to_hex(Li, Ai, Bi)
    print(f'Initial Shade Hex Code, {hex_code}')
    image = create_color_image(hex_code)
    image.show()
    L, A, B = prediction[0]
    color1 = color(Li, Ai, Bi)
    color2 = color(L, A, B)
    print(f'L: {L}, A: {A}, B: {B}')
    print('Predicted Shade: ')
    hex_code = lab_to_hex(L, A, B)
    print(f'Final Shade Hex Code, {hex_code}')
    image = create_color_image(hex_code)
    image.show()
    #lab_diff = np.sqrt(((L - Li) ** 2) + ((A - Ai) ** 2) + ((B - Bi) ** 2))
    delta_e = delta_e_cie1976(color1, color2)
    print("CIELAB Color Difference:", delta_e)


# In[ ]:


'''
Enter the number Predictions: 2
Enter Li 50.22
Enter Ai 21.47
Enter Bi 38.39
Enter Concentration 1.5
Enter pH 6
Enter Temperature 70
Enter BathRatio 0.066
Enter Duration 20
Enter Dyeing_Method SM with CuSo4
Input Shade
Initial Shade Hex Code, #a56836
L: 50.58593945703646, A: 21.692665351506832, B: 39.44277099095861
Predicted Shade: 
Final Shade Hex Code, #a66935
CIELAB Color Difference: 1.1365818511580097
Enter Li 12
Enter Ai 45
Enter Bi 46
Enter Concentration 15
Enter pH 5
Enter Temperature 70
Enter BathRatio 0.05
Enter Duration 40
Enter Dyeing_Method POM with CuSo4
Input Shade
Initial Shade Hex Code, #560000
L: 11.686373176594195, A: 45.24912602747074, B: 46.68536446804453
Predicted Shade: 
Final Shade Hex Code, #560000
CIELAB Color Difference: 0.7938198888796701'''


# In[ ]:


#63.5700	-31.7600	54.810	15.0	5.0	70	0.066	40	POM with CuSo4
#63.2700	-31.5000	55.490
#DyeingMethod_encoder.transform([Dyeing_Method])[0]

Li = 50.22
Ai = 21.47
Bi = 38.39
Concentration = 3.5
pH = 6
Temp = 70
WaterBathRatio = 0.066
Duration = 20
DyeingMethod = 'SM with CuSo4'
prediction = model.predict([[float(Li),float(Ai), float(Bi), float(Concentration), float(pH), int(Temp), float(WaterBathRatio), 
                             int(Duration), int(DyeingMethod_encoder.transform([DyeingMethod]))]])

L, A, B = prediction[0]
color2 = color(L, A, B)
color1 = color(Li, Ai, Bi)
print(f'L: {L}, A: {A}, B: {B}\n')
hex_code = lab_to_hex(L, A, B)
image = create_color_image(hex_code)
#lab_diff = np.sqrt(((L - Li) ** 2) + ((A - Ai) ** 2) + ((B - Bi) ** 2))
#print("CIELAB Color Difference:\n", lab_diff,"\n")
print("Predicted Shade")
image


# In[ ]:


hex_code = lab_to_hex(L, A, B)
image = create_color_image(hex_code)
image


# In[ ]:





# In[ ]:




