
# coding: utf-8

# # Gold Concentration Prediction Framework

# ### Load Dependencies

# In[101]:


# load packages
import pandas as pd
import numpy as np


import matplotlib.pyplot as plt
import seaborn as sns

#from fancyimpute import IterativeImputer
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split


# ### Model Creation

# In[211]:


# Predict on a set of features
def create_model(df, y_feature, x_feature):
    num_trees = 200
    num_leaves = 30
    lgbm = LGBMRegressor(n_estimators = num_trees, n_leaves = 30, importance_type='gain')
    
    print("Matrix Shape:", df.shape)
    
    # X = onehot_drop(all_data, ['Lithology1'])
    
    # df = df[~df[y_feature].isnull()]
    
    X = df.loc[:, x_feature]
    y = df.loc[:, y_feature]
    
    print('Features:', list(X))
    print('Target:', y_feature)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=80)
    
    lgbm.fit(X_train, y_train)
    
    print("Train Score:", lgbm.score(X_train, y_train))
    print("Test Score:", lgbm.score(X_test, y_test))    
    
    x_col = lgbm.feature_importances_[lgbm.feature_importances_>0]
    y_col = X.columns[lgbm.feature_importances_>0]

    plot = sns.barplot(y = y_col, x = x_col, palette = sns.color_palette("Spectral"))

    plot.set_yticklabels(['Latitude', 'Longitude', 'Depth', 'Specific Gravity', 'Magnetic Susceptibility', 'Mass In Air', 'Mass In Water'])
    #plot.set_title('Feature Importances')
    plot.set(xlabel='Information Gain')
    #plot.set_xticklabels(plot.get_xticklabels(), rotation=30)
    #plot.set(axis_bgcolor='w')
    sns.set_style("white")
    plt.show();
    #sns.set(font_scale=1);
    
    return lgbm


# ### Load Cleaned Data

# In[103]:


gold_data_final = pd.read_csv("../data/gold_data_final.csv")
gold_data_final.head()


# ### Scale Feature Values

# In[104]:


# Create copy of dataframe
gold_data_final_scaled = gold_data_final.loc[:, ['latitude', 'longitude', 'Mid_Depth', 'MassInWater', 'MagSus', 'MassInAir', 'SGFinal', 'Au']].copy()

# Scale values using log transformation to ensure high values/outliers do not affect model training
gold_data_final_scaled['Au'] = np.log((gold_data_final_scaled['Au']*100) + 1)
gold_data_final_scaled['MassInWater'] = np.log(gold_data_final_scaled['MassInWater'] + 1)
gold_data_final_scaled['MassInAir'] = np.log(gold_data_final_scaled['MassInAir'] + 1)
gold_data_final_scaled['MagSus'] = np.log(gold_data_final_scaled['MagSus'] + 105)
gold_data_final_scaled['SGFinal'] = np.log(gold_data_final_scaled['SGFinal'] + 14)

gold_data_final_scaled.describe()


# ### Create Final Gold Model

# All input features were used to generate the final model for gold concentration prediction.

# In[215]:


# Target is gold concentration
target = ['Au']
# Use these features
x_features = ['latitude', 'longitude', 'Mid_Depth', 'SGFinal', 'MagSus', 'MassInAir', 'MassInWater']

gold_model = create_model(gold_data_final_scaled, target, x_features)


# ### Start Model Creation

# Predictive features must first be generated, in order to predict the final gold concentration.

# In[106]:


gold_data_predictions = gold_data_final.loc[:, ['latitude', 'longitude', 'Mid_Depth']]


# ### Model 1: Predict MassInAir

# In[196]:


Y_FEATURE = 'MassInAir'
X_FEATURE = ['latitude', 'longitude', 'Mid_Depth']
DATAFRAME = gold_data_final_scaled
mia_model = create_model(DATAFRAME,Y_FEATURE,X_FEATURE)

gold_data_predictions['MassInAir_pred']= mia_model.predict(gold_data_predictions[X_FEATURE])


# ### Model 2: Predict Mass in Water

# In[197]:


Y_FEATURE = 'MassInWater'
X_FEATURE = ['latitude', 'longitude', 'Mid_Depth', 'MassInAir']
DATAFRAME = gold_data_final_scaled
miw_model = create_model(DATAFRAME,Y_FEATURE,X_FEATURE)

X_FEATURES_PREDICTED = ['latitude', 'longitude', 'Mid_Depth', 'MassInAir_pred']
gold_data_predictions['MassInWater_pred']= miw_model.predict(gold_data_predictions[X_FEATURES_PREDICTED])


# ### Model 3: Predict Magnetic Susceptibility

# In[198]:


Y_FEATURE = 'MagSus'
X_FEATURE = ['latitude', 'longitude', 'Mid_Depth', 'MassInWater']
DATAFRAME = gold_data_final_scaled
magsus_model = create_model(DATAFRAME,Y_FEATURE,X_FEATURE)

X_FEATURES_PREDICTED = ['latitude', 'longitude', 'Mid_Depth', 'MassInWater_pred']
gold_data_predictions['MagSus_pred']= magsus_model.predict(gold_data_predictions[X_FEATURES_PREDICTED])


# ### Model 4: Predict Specific Gravity

# In[212]:


Y_FEATURE = 'SGFinal'
X_FEATURE = ['latitude', 'longitude', 'Mid_Depth', 'MassInWater', 'MassInAir', 'MagSus']
DATAFRAME = gold_data_final_scaled
sgfinal_model = create_model(DATAFRAME,Y_FEATURE,X_FEATURE)

X_FEATURES_PREDICTED = ['latitude', 'longitude', 'Mid_Depth', 'MassInWater_pred', 'MassInAir_pred', 'MagSus_pred']
gold_data_predictions['sgfinal_pred']= sgfinal_model.predict(gold_data_predictions[X_FEATURES_PREDICTED])


# ### Create Predicted Values

# In[214]:


# Adjust figure size
plt.rcParams['figure.figsize'] = [7, 7]


# In[112]:


# Plot gold concentration predictions at every depth
def plot_predictions_iterative(df, models, cols, features):
    for idx in range(len(cols)):
        df[cols[idx]] = models[idx].predict(df[features[idx]])
        
    
    plt.contourf(lat, lon, np.array(df[cols[-1]]).reshape(1000, 1000))
    plt.show()
    
    return df

# Return an array with predicted concentration value
def array_predictions_iterative(df, models, cols, features):
    for idx in range(len(cols)):
        df[cols[idx]] = models[idx].predict(df[features[idx]])
        
    
    return np.array(df[cols[-1]]).reshape(1000, 1000)


# In[136]:


# Generate mesh grids at depths between 0 and 600
depths = {}

for mid in range(0, 601, 100):
    lat = np.arange(-30, -29, 0.001)
    lon = np.arange(135, 136, 0.001)

    xx, yy = np.meshgrid(lat, lon)
    
    xx = xx.reshape(1000000  , 1)
    yy = yy.reshape(1000000  , 1)
    
    predictions_df = pd.DataFrame(xx, columns = ['latitude'])
    predictions_df['longitude'] = yy
    predictions_df['Mid_Depth'] = mid
    
    depths[mid] = predictions_df


# In[137]:


# Round latitude and longitude values from original dataset to three decimal points
# to join mesh grid values to original site locations
gold_data_final_scaled['latitude_round'] = np.round(gold_data_final_scaled['latitude'], 3)
gold_data_final_scaled['longitude_round'] = np.round(gold_data_final_scaled['longitude'], 3)

# Grab latitude and longitude coordinates of already checked sites and make a list of them
gold_data_final_scaled['coord'] = list(zip(gold_data_final_scaled.latitude_round, 
                                           gold_data_final_scaled.longitude_round))
coordinate_list = set(gold_data_final_scaled.coord)


# In[138]:


# Populate a dictionary with concentration predictions
arrays = {}
for depth in depths.keys():
    feature_list = [['latitude', 'longitude', 'Mid_Depth'], 
                    ['latitude', 'longitude', 'Mid_Depth', 'MassInAir_pred'], 
                    ['latitude', 'longitude', 'Mid_Depth', 'MassInWater_pred'], 
                    ['latitude', 'longitude', 'Mid_Depth', 'MassInWater_pred', 'MassInAir_pred', 'MagSus_pred'], 
                    ['latitude', 'longitude', 'Mid_Depth', 'SGFinal_pred', 'MagSus_pred', 'MassInAir_pred', 'MassInWater_pred']]

    arrays[depth] = array_predictions_iterative(depths[depth], 
                                                  [mia_model, miw_model, magsus_model, sgfinal_model, gold_model], 
                                                  ['MassInAir_pred', 'MassInWater_pred', 'MagSus_pred', 'SGFinal_pred', 'Au_pred'], 
                                                  feature_list)


# In[116]:


# Create an array with averaged concentration values over all depths
final = np.zeros((1000, 1000))
for array in arrays.keys():
    final = np.add(final, arrays[array])
final = np.divide(final, 6)


# In[117]:


# Plot averaged concentration values onto a heat map
lat = np.arange(-30, -29, 0.001)
lon = np.arange(135, 136, 0.001)

lat_dict = dict(zip(np.arange(1000), lat.T))
lon_dict = dict(zip(np.arange(1000), lon.T))

plt.contourf(lat, lon, final)
plt.show()


# In[134]:


# From the averaged array, find the max n values
n = 200
indices = np.argpartition(final.flatten(), -2)[-n:]
max_values_df = pd.DataFrame(np.vstack(np.unravel_index(indices, final.shape)).T, columns = ['lat_ind', 'lon_ind'])

max_values_df['lat'] = max_values_df['lat_ind'].apply(lambda x: lat_dict[x])
max_values_df['lon'] = max_values_df['lon_ind'].apply(lambda x: lon_dict[x])

# Check if highest predicted coordinates match with any existing sites
max_values_df['coord'] = list(zip(np.round(max_values_df.lat, 3), np.round(max_values_df.lon, 3)))
max_values_df['matching'] = max_values_df['coord'].apply(lambda x: x in coordinate_list)
display(max_values_df[max_values_df['matching']])
max_values_df['conc_log'] = final[max_values_df.lat_ind, max_values_df.lon_ind]

max_values_df['conc'] = (np.exp(max_values_df['conc_log']) - 1)/100

max_values_df = max_values_df.sort_values('conc', ascending = False).reset_index()
max_values_df.head()


# In[133]:


# Save predicted results to csv
max_values_df.to_csv("../data/top_gold_predictions_200_final.csv")
max_values_df[:10].to_csv("../data/final_gold_submission.csv")


# In[ ]:


for depth in depths.keys():
    feature_list = [['latitude', 'longitude', 'Mid_Depth'], 
                    ['latitude', 'longitude', 'Mid_Depth', 'MassInAir_pred'], 
                    ['latitude', 'longitude', 'Mid_Depth', 'MassInWater_pred'], 
                    ['latitude', 'longitude', 'Mid_Depth', 'MassInWater_pred', 'MassInAir_pred', 'MagSus_pred'], 
                    ['Mid_Depth', 'SGFinal_pred', 'MagSus_pred', 'MassInAir_pred', 'MassInWater_pred']]

    gold_predictions = plot_predictions_iterative(depths[depth], 
                                                  [mia_model, miw_model, magsus_model, sgfinal_model, gold_model], 
                                                  ['MassInAir_pred', 'MassInWater_pred', 'MagSus_pred', 'SGFinal_pred', 'Au_pred'], 
                                                  feature_list)


# In[144]:


combined_frame = pd.DataFrame()

for keys in depths.keys():
    print(keys)
    #display(depths[keys].sort_values('Au_pred', ascending = False)[:10])
    combined_frame = pd.concat([combined_frame, depths[keys]])
    
#display(combined_frame.head())


# In[147]:


combined_frame_sorted = combined_frame.sort_values('Au_pred', ascending = False)
combined_frame_sorted['Au_pred_raw'] = (np.exp(combined_frame_sorted['Au_pred']) -1)/100
combined_frame_sorted.head()


# In[146]:


len(combined_frame)

