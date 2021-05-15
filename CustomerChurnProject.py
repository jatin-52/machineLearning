
# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# In[2]:

path = 'C:/Jatin/code/MachineLearning/'
data = pd.read_excel(path + "WA_Fn-UseC_-Telco-Customer-Churn.csv.xlsx")


# ### Summary of data

# In[3]:


data.head()


# In[4]:


data.info()


# ### Checking for NULL data

# In[5]:


data.isna().sum()


# ### Checking for duplicated data

# In[6]:


data.duplicated().sum()


# ### Number of Churn

# In[7]:


sns.set(style="white", palette="deep", color_codes=True)
sns.despine(left=True)
sns.countplot(data["Churn"]);


# In[8]:


plt.pie(data["Churn"].value_counts(),explode=(0,0.1), autopct='%1.1f%%',
        shadow=True, startangle=90,labels=data["Churn"].unique())
plt.axis('equal') ;


# ### Data cleaning

# In[9]:


data.query("TotalCharges == ' '").TotalCharges.count()


# In[10]:


data["TotalCharges"] = data["TotalCharges"].replace(" ",np.nan)
data.dropna(inplace = True);


# In[11]:


data["TotalCharges"] = data["TotalCharges"].astype("float")


# In[12]:


data.info()


# In[13]:


data[data["TotalCharges"]<0]["TotalCharges"].count()


# In[14]:


temp_columns = [col for col in data.columns if col not in ("customerID","gender","MonthlyCharges","TotalCharges","Churn")]


# In[15]:


temp_columns


# In[16]:


for col in temp_columns:
    print("{} : {}".format(col,data[col].unique()))


# In[17]:


for col in temp_columns:
    if col in ("OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies"):
        data[col] = data[col].replace({'No internet service':'No'})


# In[18]:


temp_tenure = np.array(data["tenure"].tolist())
print("min: {}".format(temp_tenure.min()))
print("max: {}".format(temp_tenure.max()))


# In[19]:


def tenure_to_group(data):
    if data["tenure"] <=12:
        return "0_1_year"
    elif (data["tenure"] > 12) & (data["tenure"] <= 24 ):
        return "1_2_year"
    elif (data["tenure"] > 24) & (data["tenure"] <= 36) :
        return "2_3_year"
    elif (data["tenure"] > 36) & (data["tenure"] <= 48) :
        return "3_4_year"
    elif data["tenure"] > 48 & (data["tenure"] <= 60):
        return "4_5_year"
    elif data["tenure"] > 60 & (data["tenure"] <= 72):
        return "5_6_year"
data["Tenure_Group"] = data.apply(lambda data:tenure_to_group(data),axis = 1)
sns.countplot(data["Tenure_Group"]);


# ### Visualization

# In[20]:


f, axes = plt.subplots(figsize=(18, 8))
sns.countplot(data["tenure"],hue = data["Churn"]);


# In[21]:


f, axes = plt.subplots(nrows=6, ncols=3, figsize=(20, 30))

sns.countplot(data["Churn"],hue = data["gender"],ax = axes[0,0])
sns.countplot(data["Churn"],hue = data["SeniorCitizen"],ax = axes[0,1])
sns.countplot(data["Churn"],hue = data["Partner"],ax = axes[0,2])
sns.countplot(data["Churn"],hue = data["Dependents"],ax = axes[1,0])
sns.countplot(data["Churn"],hue = data["PhoneService"],ax = axes[1,1])
sns.countplot(data["Churn"],hue = data["MultipleLines"],ax = axes[1,2])
sns.countplot(data["Churn"],hue = data["InternetService"],ax = axes[2,0])
sns.countplot(data["Churn"],hue = data["OnlineSecurity"],ax = axes[2,1])
sns.countplot(data["Churn"],hue = data["OnlineBackup"],ax = axes[2,2])
sns.countplot(data["Churn"],hue = data["DeviceProtection"],ax = axes[3,0])
sns.countplot(data["Churn"],hue = data["TechSupport"],ax = axes[3,1])
sns.countplot(data["Churn"],hue = data["StreamingTV"],ax = axes[3,2])
sns.countplot(data["Churn"],hue = data["StreamingMovies"],ax = axes[4,0])
sns.countplot(data["Churn"],hue = data["Contract"],ax = axes[4,1])
sns.countplot(data["Churn"],hue = data["PaperlessBilling"],ax = axes[4,2])
sns.countplot(data["Churn"],hue = data["PaymentMethod"],ax = axes[5,0])
sns.countplot(data["Churn"],hue = data["Tenure_Group"],ax = axes[5,1])
sns.countplot(data["Tenure_Group"],ax = axes[5,2]);

plt.setp(axes, yticks=[])
plt.tight_layout()


# In[22]:


f, axes = plt.subplots( ncols=3, figsize=(20, 5))
sns.boxplot(x="Churn", y="tenure", data=data,palette='rainbow',ax = axes[0]);
sns.boxplot(x="Churn", y="MonthlyCharges", data=data,palette='rainbow',ax = axes[1])
sns.boxplot(x="Churn", y="TotalCharges", data=data,palette='rainbow',ax = axes[2])


# In[23]:


temp_cols = data.drop("SeniorCitizen",axis = 1)
sns.pairplot(temp_cols,hue='Churn',palette='rainbow')


# In[24]:


sns.lmplot(x = "MonthlyCharges", y= "TotalCharges", data=data,fit_reg = False,hue = "Tenure_Group",aspect=12/9);


# In[25]:


f, axes = plt.subplots( ncols=2, figsize=(20, 6))
sns.barplot(x='Tenure_Group',y='MonthlyCharges',data=data,hue = "Churn",ax = axes[0])
sns.barplot(x='Tenure_Group',y='TotalCharges',data=data,hue = "Churn",ax = axes[1])


# In[26]:


data.info()


# ### Data preprocessing

# In[27]:


cat_cols = [x for x in data.columns if data[x].nunique()<6 and x!="Churn"]
num_cols = [x for x in data.columns if data[x].nunique()>6 and x!="customerID"]
id_customer = data["customerID"]
label = data["Churn"]
label = label.apply(lambda x: 1 if x == "Yes" else 0)


# In[28]:


from sklearn.preprocessing import MinMaxScaler

features_log_transformed = pd.DataFrame(data = data[num_cols])
features_log_transformed[num_cols] = data[num_cols].apply(lambda x: np.log(x + 1))

scaler = MinMaxScaler()
features_log_minmax_transform = pd.DataFrame(data = features_log_transformed)
features_log_minmax_transform[num_cols] = scaler.fit_transform(features_log_transformed[num_cols])


# In[29]:


sns.heatmap(features_log_minmax_transform.corr(),annot=True,cmap='jet');


# In[30]:


features_log_minmax_transform.drop("tenure",inplace = True, axis = 1)


# In[31]:


data.drop(["MonthlyCharges","TotalCharges","tenure"],axis = 1, inplace = True)
data = pd.concat([data, features_log_minmax_transform], axis=1)


# In[32]:


data.info()


# In[33]:


data.isna().sum()


# In[34]:


data.duplicated().sum()


# In[35]:


data.drop("Churn",inplace = True, axis = 1)
data.drop("customerID",inplace = True, axis = 1)


# In[36]:


data.info()


# In[37]:


data = pd.get_dummies(data = data,columns = cat_cols)


# In[38]:


data.head()


# In[39]:


data_original = pd.concat([data, label,id_customer], axis=1)


# In[40]:


data_original.info()


# In[41]:


data_original.head()


# ### Evaluate Algorithms

# In[42]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data, label, test_size = 0.3, random_state = 42)

print("Training set has {} samples.".format(X_train.shape[0]))
print("Testing set has {} samples.".format(X_test.shape[0]))


# In[43]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score,roc_curve

def apply_classifier(clf,xTrain,xTest,yTrain,yTest):
    
    clf.fit(xTrain, yTrain)
    predictions = clf.predict(xTest)
    conf_mtx = confusion_matrix(yTest,predictions)
    f, axes = plt.subplots(ncols=2, figsize=(15, 5))
    sns.heatmap(conf_mtx,annot=True,cmap='tab20c',cbar = False,fmt = "g",ax = axes[0])
    axes[0].set_xlabel('Predicted labels')
    axes[0].set_ylabel('True labels')
    axes[0].set_title('Confusion Matrix'); 
    axes[0].xaxis.set_ticklabels(['Not Churn', 'Churn'])
    axes[0].yaxis.set_ticklabels(['Not Churn', 'Churn'])

    print("\n Classification report : \n {}".format(classification_report(yTest,predictions)))
    
    roc_auc = roc_auc_score(yTest,predictions) 
    print ("Area under ROC curve : ",roc_auc,"\n")

    fpr, tpr,_ = roc_curve(yTest, predictions)
    axes[1].plot(fpr,tpr,label= "auc="+str(roc_auc));
    axes[1].plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")



# In[44]:


decision_tree = DecisionTreeClassifier(random_state = 42);
apply_classifier(decision_tree,X_train, X_test, y_train, y_test)


# In[45]:


logistic_reg = LogisticRegression(random_state = 42)
apply_classifier(logistic_reg,X_train, X_test, y_train, y_test)


# In[46]:


svm_model = SVC(random_state = 42)
apply_classifier(svm_model,X_train, X_test, y_train, y_test)


# In[47]:


random_forest = RandomForestClassifier(random_state = 42)
apply_classifier(random_forest,X_train, X_test, y_train, y_test)


# In[48]:




# ### Tuning Parameters

# In[49]:


Tree_parameters = {"max_depth": [3,4,5,6],
                   "min_samples_leaf":[1,2,3,4]}

LogReg_parameters = {
    "C":[0.25,0.5,0.75,1.0,1.5,2.0,2.5,3.0,4.0,10.0],
    "solver":["newton-cg", "lbfgs", "sag", "saga"],
    "tol":[0.01,0.001,0.0001,0.00001],
    "warm_start":["True","False"]}

SVM_parameters = {
    "C":[1.0,2.0,3.0],
    "cache_size":[100,200],
    "decision_function_shape":['ovo','ovr'],
    "kernel":['sigmoid',"linear"],
    "tol":[0.001,0.0001]}

RandomForest_parameters = {
    "n_estimators" :[10,15,20,25,30], 
    "criterion": ["entropy","gini"],
    "max_depth" : [5,10,15],
    "min_samples_split":[2,4,8,16],
    "max_features":["sqrt","auto","log2"],
    "class_weight" : ["balanced_subsample","balanced"]}

Xgboost_parameters = {"max_depth" : [3,4,5,6],
    "learning_rate" : [0.001,0.0001],
    "booster" : ["gbtree","gblinear","dart"],
    "min_child_weight" : [1,2,3,4]
                     
                     }


# In[50]:


from sklearn.grid_search import GridSearchCV

def grid_search(clf,parameters,xTrain,Ytrain):
    
    
    grid_obj = GridSearchCV(clf,parameters,scoring = 'roc_auc',cv = 5)
    grid_fit = grid_obj.fit(xTrain,Ytrain)
    best_clf = grid_fit.best_estimator_

    return best_clf


# In[51]:


tree_grid = grid_search(decision_tree,Tree_parameters,X_train,y_train);
apply_classifier(tree_grid,X_train, X_test, y_train, y_test)


# In[52]:


logReg_grid = grid_search(logistic_reg,LogReg_parameters,X_train,y_train);
apply_classifier(logReg_grid,X_train, X_test, y_train, y_test)


# In[53]:


svm_grid = grid_search(svm_model,SVM_parameters,X_train,y_train);
apply_classifier(svm_grid,X_train, X_test, y_train, y_test)


# In[54]:


randomForest_grid = grid_search(random_forest,RandomForest_parameters,X_train,y_train);
apply_classifier(randomForest_grid,X_train, X_test, y_train, y_test)


# In[55]:


xgBoost_grid = grid_search(xg_boost,Xgboost_parameters,X_train,y_train);
apply_classifier(xgBoost_grid,X_train, X_test, y_train, y_test)


# In[56]:


from sklearn.ensemble import AdaBoostClassifier

model = AdaBoostClassifier(base_estimator = randomForest_grid, n_estimators = 4)
apply_classifier(model,X_train, X_test, y_train, y_test)


# ### Upsampling

# In[57]:


from sklearn.utils import resample

upsample_data = data_original

majority = upsample_data[upsample_data["Churn"]==0]
minority = upsample_data[upsample_data["Churn"]==1]

minority_upsampled = resample(minority, replace=True, n_samples=5163,random_state=42) 
del(upsample_data)
upsample_data = pd.concat([majority,minority_upsampled])


# In[58]:


sns.countplot(upsample_data["Churn"]);


# In[59]:


id_customer_upsample = upsample_data["customerID"]
label_upsample = upsample_data["Churn"]
upsample_data.drop("Churn",inplace = True, axis = 1)
upsample_data.drop("customerID",inplace = True, axis = 1)


# In[60]:


from sklearn.cross_validation import train_test_split

X_train_upS, X_test_upS, y_train_upS, y_test_upS = train_test_split(upsample_data, label_upsample, test_size = 0.3, random_state = 42)

print("Training set has {} samples.".format(X_train_upS.shape[0]))
print("Testing set has {} samples.".format(X_test_upS.shape[0]))


# In[61]:


decision_tree = DecisionTreeClassifier(random_state = 42);
apply_classifier(decision_tree,X_train_upS, X_test_upS, y_train_upS, y_test_upS)

# data for flask
import joblib 
joblib.dump(decision_tree, path + '/model_DecisionTree.pkl')
print("Model dumped!")

model_columns = list(X_train.columns)
print(model_columns)
joblib.dump(model_columns, path + '/model_columns_DecisionTree.pkl')
print("Models columns dumped!")


# In[62]:


logistic_reg = LogisticRegression(random_state = 42)
apply_classifier(logistic_reg,X_train_upS, X_test_upS, y_train_upS, y_test_upS)


# In[63]:


svm_model = SVC(random_state = 42)
apply_classifier(svm_model,X_train_upS, X_test_upS, y_train_upS, y_test_upS)


# In[64]:


random_forest = RandomForestClassifier(random_state = 42)
apply_classifier(random_forest,X_train_upS, X_test_upS, y_train_upS, y_test_upS)



# ### Final Model

# In[66]:


model = AdaBoostClassifier(base_estimator = random_forest, n_estimators = 4)
apply_classifier(model,X_train_upS, X_test_upS, y_train_upS, y_test_upS)


# In[67]:


importances = pd.DataFrame({'feature':data.columns,'importance':np.round(model.feature_importances_,3)})
importances = importances.sort_values('importance',ascending=False).set_index('feature')
importances[0:7].plot.bar(figsize=(20, 8))


# In[ ]:




