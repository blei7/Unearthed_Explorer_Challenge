
# coding: utf-8

# # Copper Concentration Prediction Framework

# ### Load Dependencies

# In[1]:


# load packages
import pandas as pd
import numpy as np


import matplotlib.pyplot as plt
import seaborn as sns

#from fancyimpute import IterativeImputer
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split


# ### Model Creation

# In[70]:


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

    plot.set_yticklabels(['Latitude', 'Longitude', 'Depth', 'Sulfur Concentration', 'Cerium Concentration'])
    #plot.set_title('Feature Importances')
    plot.set(xlabel='Information Gain')
    #plot.set_xticklabels(plot.get_xticklabels(), rotation=30)
    #plot.set(axis_bgcolor='w')
    sns.set_style("white")
    plt.show();
    #sns.set(font_scale=1);
    
    return lgbm


# ### Cleaned Data Load

# In[3]:


copper_data_final = pd.read_csv("../data/copper_data_final.csv")
copper_data_final.head()


# In[4]:


copper_data_final.describe()


# ### Scale Copper Values

# In[8]:


# Create a copy of the input dataframe
copper_data_final_scaled = copper_data_final.loc[:, ['latitude', 'longitude', 'Mid_Depth', 'La', 'Ce', 'S', 'MassInWater', 'MassInAir', 'MagSus', 'SGFinal', 'Cu']].copy()

# Scale values using log transformation to ensure high values/outliers do not affect model training
copper_data_final_scaled['La'] = np.log((copper_data_final['La']) + 40)
copper_data_final_scaled['Ce'] = np.log((copper_data_final['Ce']) + 14)
copper_data_final_scaled['MassInWater'] = np.log((copper_data_final['MassInWater']) + 5026)
copper_data_final_scaled['MassInAir'] = np.log((copper_data_final['MassInAir']) + 11352)
copper_data_final_scaled['MagSus'] = np.log((copper_data_final['MagSus']) + 244)
copper_data_final_scaled['SGFinal'] = np.log((copper_data_final['SGFinal']) + 16)
copper_data_final_scaled['Cu'] = np.log((copper_data_final['Cu']) + 1)

copper_data_final_scaled.describe()


# ### Create Final Copper Model

# In[71]:


# Target is copper concentration
target = ['Cu']
# Use these features
x_features = ['latitude', 'longitude', 'Mid_Depth', 'S', 'Ce']

copper_model = create_model(copper_data_final_scaled, target, x_features)


# ### Start Model Creation

# Predictive features must first be generated, in order to predict the final gold concentration.

# In[28]:


copper_data_predictions = copper_data_final.loc[:, ['latitude', 'longitude', 'Mid_Depth']]


# ### Model 1: Predict Ce

# In[34]:


Y_FEATURE = 'Ce'
X_FEATURE = ['latitude', 'longitude', 'Mid_Depth']
DATAFRAME = copper_data_final_scaled
ce_model = create_model(DATAFRAME,Y_FEATURE,X_FEATURE)

X_FEATURES_PREDICTED = ['latitude', 'longitude', 'Mid_Depth']
copper_data_predictions['ce_pred']= ce_model.predict(copper_data_predictions[X_FEATURES_PREDICTED])


# ### Model 2: Predict S

# In[35]:


Y_FEATURE = 'S'
X_FEATURE = ['latitude', 'longitude','Mid_Depth', "Ce"]
DATAFRAME = copper_data_final
s_model = create_model(DATAFRAME,Y_FEATURE,X_FEATURE)

X_FEATURES_PREDICTED = ['latitude', 'longitude', 'Mid_Depth', 'ce_pred']
copper_data_predictions['s_pred']= s_model.predict(copper_data_predictions[X_FEATURES_PREDICTED])


# ### Create Predicted Values

# In[64]:


# Adjust figure size
plt.rcParams['figure.figsize'] = [7, 7]


# In[39]:


def plot_predictions_iterative(df, models, cols, features):
    for idx in range(len(cols)):
        df[cols[idx]] = models[idx].predict(df[features[idx]])
        
    # df['raw_conc'] = np.exp(df[cols[-1]]) - 1
    # print(df.describe())
    plt.contourf(lat, lon, np.array(df[cols[-1]]).reshape(1000, 1000))
    plt.show()
    
    return df

def array_predictions_iterative(df, models, cols, features):
    for idx in range(len(cols)):
        df[cols[idx]] = models[idx].predict(df[features[idx]])
        
    
    return np.array(df[cols[-1]]).reshape(1000, 1000)


# In[38]:


depths = {}

for mid in range(0, 601, 100):
    lat = np.arange(-30, -29, 0.001)
    lon = np.arange(135, 136, 0.001)

    xx, yy = np.meshgrid(lat, lon)
    
    xx = xx.reshape(1000000 , 1)
    yy = yy.reshape(1000000 , 1)
    
    predictions_df = pd.DataFrame(xx, columns = ['latitude'])
    predictions_df['longitude'] = yy
    predictions_df['Mid_Depth'] = mid
    
    depths[mid] = predictions_df


# In[41]:


arrays = {}
for depth in depths.keys():
    feature_list = [['latitude', 'longitude', 'Mid_Depth'], 
                    ['latitude', 'longitude', 'Mid_Depth', "Ce_pred"], 
                    ['latitude', 'longitude', 'Mid_Depth', 'Ce_pred', 'S_pred']]

    arrays[depth] = array_predictions_iterative(depths[depth], 
                                                  [ce_model, s_model, copper_model], 
                                                  ['Ce_pred', 'S_pred', 'Cu_pred'], 
                                                  feature_list)
    


# In[ ]:


# Round latitude and longitude values from original dataset to three decimal points
# to join mesh grid values to original site locations
copper_data_final_scaled['latitude_round'] = np.round(copper_data_final_scaled['latitude'], 3)
copper_data_final_scaled['longitude_round'] = np.round(copper_data_final_scaled['longitude'], 3)

# Grab latitude and longitude coordinates of already checked sites and make a list of them
copper_data_final_scaled['coord'] = list(zip(copper_data_final_scaled.latitude_round, 
                                           copper_data_final_scaled.longitude_round))


coordinate_list = set(copper_data_final_scaled.coord)


# In[42]:


# Create an array with averaged concentration values over all depths
final = np.zeros((1000, 1000))
for array in arrays.keys():
    final = np.add(final, arrays[array])
final = np.divide(final, 6)


# In[43]:


# Plot averaged concentration values onto a heat map
lat = np.arange(-30, -29, 0.001)
lon = np.arange(135, 136, 0.001)

lat_dict = dict(zip(np.arange(1000), lat.T))
lon_dict = dict(zip(np.arange(1000), lon.T))

plt.contourf(lat, lon, final)
plt.show()


# In[52]:


# From the averaged array, find the max n values
n = 200
indices = np.argpartition(final.flatten(), -2)[-n:]
max_values_df = pd.DataFrame(np.vstack(np.unravel_index(indices, final.shape)).T, columns = ['lat_ind', 'lon_ind'])

max_values_df['lat'] = max_values_df['lat_ind'].apply(lambda x: lat_dict[x])
max_values_df['lon'] = max_values_df['lon_ind'].apply(lambda x: lon_dict[x])
max_values_df['coord'] = list(zip(np.round(max_values_df.lat, 3), np.round(max_values_df.lon, 3)))
max_values_df['matching'] = max_values_df['coord'].apply(lambda x: x in coordinate_list)
display(max_values_df[max_values_df['matching']])
max_values_df['conc_log'] = final[max_values_df.lat_ind, max_values_df.lon_ind]

max_values_df['conc'] = (np.exp(max_values_df['conc_log']) - 1)/100
max_values_df = max_values_df.sort_values('conc', ascending = False).reset_index()
max_values_df.head()


# In[53]:


# Save predicted results to csv
max_values_df.to_csv("../data/top_copper_predictions_200_final.csv")
max_values_df[:10].to_csv("../data/final_copper_submission.csv")


# In[79]:


for depth in depths.keys():
    feature_list = [['latitude', 'longitude', 'Mid_Depth'], 
                    ['latitude', 'longitude', 'Mid_Depth'], 
                    ['latitude', 'longitude', 'Mid_Depth', 'MassInWater_pred'], 
                    ['latitude', 'longitude', 'Mid_Depth', "Ce_pred"], 
                    ['latitude', 'longitude', 'Mid_Depth', 'Ce_pred', 'S_pred', 'MassInWater_pred', 'MagSus_pred']]

    copper_predictions = plot_predictions_iterative(depths[depth], 
                                                  [magsus_model, massinwater_model, ce_model, s_model, copper_model], 
                                                  ['MagSus_pred', 'MassInWater_pred', 'Ce_pred', 'S_pred', 'Cu_log_pred'], 
                                                  feature_list)
    


# In[54]:


combined_frame = pd.DataFrame()

for keys in depths.keys():
    print(keys)
    #display(depths[keys].sort_values('Au_pred', ascending = False)[:10])
    combined_frame = pd.concat([combined_frame, depths[keys]])
    
#display(combined_frame.head())


# In[57]:


combined_frame_sorted = combined_frame.sort_values('Cu_pred', ascending = False)
combined_frame_sorted['Cu_pred_raw'] = (np.exp(combined_frame_sorted['Cu_pred']) -1)/100
combined_frame_sorted.head()

