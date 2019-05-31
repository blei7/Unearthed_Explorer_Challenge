
# coding: utf-8

# In[1]:


#-------------------------------------------------------------------#
                        # Load Dependencies #
#-------------------------------------------------------------------#


# load packages
import pandas as pd
import numpy as np

from itertools import compress
import matplotlib.pyplot as plt
import seaborn as sns

from fancyimpute import IterativeImputer
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.linear_model import Ridge

path='../drilling_database/drilling_database'


#-------------------------------------------------------------------#
                    # Data Load and Cleaning #
#-------------------------------------------------------------------#


#load assays

assays=pd.read_csv(path+'/assays_cleaned.csv')
assays=assays.drop('Unnamed: 0',1)
assays=assays.rename(columns={'SITE_ID':'SiteId'})
print("Rows in Assays",len(assays))


for col_units in list(assays):
    if "UNITS" in col_units:
        #print(col_units)
        col=col_units.split("_")[0]
        
        
        assays[col_units]=assays[col_units].astype(str).str.lower()
        
        # chnage ppb to ppm
        assays.loc[assays[col_units]=='ppb', [col]] /= 1000
        
        # quantities in percent make nan
        assays.loc[assays[col_units]=='percent', [col]] = np.nan
        
        # quantities less than 0 
        assays[col]=assays[col].mask(assays[col].lt(0),0)
        
print("Rows in Assays after cleaning",len(assays))

#-------------------------------------------------------------------#
                # Feature Generation: Mid Depth #
#-------------------------------------------------------------------#

assays['Mid_Depth']=(assays['DEPTH_FROM']+assays['DEPTH_TO'])/2
assays.head()


# creating site depth data
site_depth=assays[['SiteId','Mid_Depth']].drop_duplicates()
print("Rows in Site Depth",len(site_depth))

# collars
collars=pd.read_csv(path+'/collars.csv')
collars=collars[['SiteId','latitude','longitude']]
print("Rows in Collars",len(collars))


# magnet
magnet=pd.read_csv(path+'/magnetic_susceptibility.csv')
magnet=magnet[['SiteID','MagSus', 'DepthFrom','DepthTo']]
magnet=magnet.rename(columns={'DepthFrom':'DepthFrom_Mag','DepthTo':'DepthTo_Mag'})
print("Rows in Magnet",len(magnet))
magnet.head()

# gravity
gravity=pd.read_csv(path+'/specific_gravity.csv')
gravity=gravity[['SiteID','MassInAir','MassInWater','SGFinal', 'DepthFrom','DepthTo']]
gravity=gravity.rename(columns={'DepthFrom':'DepthFrom_Mag','DepthTo':'DepthTo_Mag'})
print("Rows in Gravity",len(gravity))
gravity.head()


# In[2]:


#-------------------------------------------------------------------#
                    # Merging Datasets
#-------------------------------------------------------------------#

# Merge Assays and Collars #
assays_collars=pd.merge(collars,assays,how='inner', on=['SiteId'])
print("Rows after Assays and Collars merge",len(assays_collars))


# merge site_depth with mag
mag_site=pd.merge(magnet,site_depth,how='left', left_on=['SiteID'], right_on=['SiteId']).drop('SiteID',1)
mag_site=mag_site[['SiteId','Mid_Depth','MagSus']][(mag_site['Mid_Depth']>=mag_site['DepthFrom_Mag']) & (mag_site['Mid_Depth']<=mag_site['DepthTo_Mag'])]
print("Rows after Magnet and Site Depth merge",len(mag_site))

# Merge Assays_Collars And Magnetic Susceptibility #
assays_collars_mag=pd.merge(assays_collars,mag_site,how='left', on=['SiteId','Mid_Depth'])
print("Rows after Assays_Collars And Magnetic Susceptibility merge",len(assays_collars_mag))
assays_collars_mag.head()

# merge site_depth with mag
gravity_site=pd.merge(gravity,site_depth,how='left', left_on=['SiteID'], right_on=['SiteId']).drop('SiteID',1)
gravity_site=gravity_site[['SiteId','MassInAir','MassInWater','SGFinal','Mid_Depth']][(gravity_site['Mid_Depth']>=gravity_site['DepthFrom_Mag']) & (gravity_site['Mid_Depth']<=gravity_site['DepthTo_Mag'])]
print("Rows after Magnet and Site Depth merge",len(gravity_site))

# Merge Assays_Collars_Magnetic And Gravity #
assays_collars_mag_gravity=pd.merge(assays_collars_mag,gravity_site,how='left',on=['SiteId','Mid_Depth'])
print("Rows after Assays_Collars_Magnetic And Gravity merge",len(assays_collars_mag_gravity))
assays_collars_mag_gravity.head()
assays_collars_mag_gravity=assays_collars_mag_gravity.drop('SiteId',1)


# In[4]:


#-------------------------------------------------------------------#
              # Feature Selection: Correlated Elements #
#-------------------------------------------------------------------#

def corr_features(data_df,element,threshold=0.3):

    # correlation matrix

    data_df=data_df._get_numeric_data()
    rows, cols = data_df.shape
    flds = list(data_df.columns)
    corr = data_df.corr().values
    index=['column_x','column_y','corr']
    corr_df=pd.DataFrame(columns=index)
    for i in range(cols):
        for j in range(i+1, cols):
            if corr[i,j] > threshold:
                temp=flds[i], flds[j], corr[i,j]
                corr_df=corr_df.append(pd.Series(temp,index=index),ignore_index=True)
                
    corr_df=corr_df[(corr_df['column_x']==element)|(corr_df['column_y']==element)]
    cols=set(corr_df['column_x']).union(corr_df['column_y']).union({element})
    return list(cols)

element_features_Cu=corr_features(assays,"Cu",0.3)
print("Elements as features for Copper",element_features_Cu)

element_features_Au=corr_features(assays,"Au",0.25)
print("Elements as features for Gold",element_features_Au)


# In[5]:


#-------------------------------------------------------------------#
            # Feature Set for Copper and Gold #
#-------------------------------------------------------------------#

geo_features= ['latitude','longitude','DEPTH_FROM','DEPTH_TO','Mid_Depth','MagSus','MassInAir','MassInWater','SGFinal']


Copper_Data=assays_collars_mag_gravity[geo_features+element_features_Cu]
Copper_Data=Copper_Data[Copper_Data['Cu'].isnull()==0].reset_index(drop=True)
Copper_Features=list(set(Copper_Data)-{"Cu"})

Gold_Data=assays_collars_mag_gravity[geo_features+element_features_Au]
Gold_Data=Gold_Data[Gold_Data['Au'].isnull()==0].reset_index(drop=True)
Gold_Features=list(set(Gold_Data)-{"Au"})


# In[6]:


#-------------------------------------------------------------------#
                    # Missing Value Imputation #
#-------------------------------------------------------------------#

def impute(df,cols):
    
    my_imputer = IterativeImputer()

    # fit
    print("Input shape of data",df[cols].shape)
    my_imputer.fit(df[cols])
    

    # transform
    df_filled = my_imputer.transform(df[cols])
    print("Output shape of data",df[cols].shape)
    
    df=pd.DataFrame(df_filled,columns=cols)
    return df

Imputed_Copper_Data=impute(Copper_Data,Copper_Features)
Imputed_Gold_Data=impute(Gold_Data,Gold_Features)


# In[7]:


#-------------------------------------------------------------------#
            # Possible features for Copper and Gold #
#-------------------------------------------------------------------#

def feature_selection(X,y):
    
    Xtrain, Xvalidate, ytrain, yvalidate = train_test_split(X, y, test_size=0.20, random_state=80)

    rfe_scores=[]
    features=range(1,Xtrain.shape[1]+1)
    # looping over number of features
    for i in features:
        rfe = RFE(estimator = Ridge(), n_features_to_select = i)
        rfe.fit(Xtrain,ytrain)
        Xtrain_new=Xtrain.iloc[:,rfe.support_]
        Xvalidate_new=Xvalidate.iloc[:,rfe.support_]
        lr=LGBMRegressor(n_estimators = 200, n_leaves = 30, importance_type='gain')
        lr.fit(Xtrain_new, ytrain)
        rfe_scores.append(round((1-lr.score(Xvalidate_new, yvalidate))*100,2))
    plt.plot(features,rfe_scores,label="score",linestyle='--', marker='o', color='b')
    plt.xlabel("Degree")
    plt.ylabel("Validation Loss")
    plt.legend()
    plt.show()
    print("The validation error is lowest when features=",features[np.argmin(rfe_scores)])
    
    
    rfe = RFE(estimator = Ridge(), n_features_to_select = features[np.argmin(rfe_scores)])
    rfe.fit(Xtrain,ytrain)
    features_selected=list(compress(list(X),rfe.support_))
    print(features_selected)
    return features_selected


Copper_Selected_Features=feature_selection(Imputed_Copper_Data,Copper_Data['Cu'])
Gold_Selected_Features=feature_selection(Imputed_Gold_Data,Gold_Data['Au'])


# In[9]:


# Export Datasets
Copper_Data_Final=pd.merge(Imputed_Copper_Data,Copper_Data['Cu'], left_index=True, right_index=True)
Gold_Data_Final=pd.merge(Imputed_Gold_Data,Gold_Data['Au'], left_index=True, right_index=True)

Copper_Data_Final.to_csv('../data/copper_data_final.csv')
Gold_Data_Final.to_csv('../data/gold_data_final.csv')

