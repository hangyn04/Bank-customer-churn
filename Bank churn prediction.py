import numpy as np     #Linear Algebra
import pandas as pd    #data processing, CSV file I/O
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.metrics import confusion_matrix,roc_curve,roc_auc_score

from imblearn.over_sampling import SMOTE

### data loading
df = pd.read_csv('C:/Users/Ha/Downloads/Churn_Modelling(1).csv')
#print(df.head())
#print(df.info())
#print(df.columns)

## Missing values handling
#print(df.isnull().sum().to_frame().rename(columns={0:"Total of missing values"}))

#Check the cardinality of categorical cols
#print(df.describe(include='object'))
#print(df.sample(5))

#Dropping insignificant features 
# df.drop(columns=['RowNumber','CustomerId','Surname'],inplace=True)
# # #print(df.sample(5))

# # #Rename Exited col's value with no and yes
# # df['Exited'].replace({0:'No',1:'Yes'},inplace=True)
# # #print(df.head())

# # # Categorize data from age into age group col
# # #### Define the bins and corresponding labels
# bins = [0, 20, 30, 40, 50, 60, 70, float('inf')]  # Bins for age ranges
# labels = ['0-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71+']  # Labels for the ranges

# # # # Create the AgeGroup column
# df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels, right=True)

# # # # Drop the original Age column
# df = df.drop('Age', axis=1)

# # # Categorize data from CreditScore into specific group col
# bins = [0, 400, 500, 600, 700, 800, float('inf')]  # Bins for age ranges
# labels = ['0-400', '401-500', '501-600', '601-700', '701-800', '800+']

# # # Create the Credit Score column
# df['Credit Score'] = pd.cut(df['CreditScore'], bins=bins, labels=labels, right=True)

# # # Drop the original Age column
# df = df.drop('CreditScore', axis=1)

# # Categorize data from Balance into specific group col
# bins = [-float('inf'), 0, 1000, 10000, 100000, 200000, float('inf')]  # Bins to include 0 as a separate category
# labels = ['0', '0-1000', '1000-10000', '10000-100000', '100000-200000', '200000+']  # Labels corresponding to bins

# # # Create the Acct Balance column
# df['Acct Balance'] = pd.cut(df['Balance'], bins=bins, labels=labels, right=True)

# # # Drop the original Balance column
# df = df.drop('Balance', axis=1)

# print(df.head())
# #print(df.info())
# # Save the DataFrame to the existing CSV file
#df.to_csv('C:/Users/Ha/Downloads/Churn_Modelling.csv', index=False)

# make a bar plot to see churn rate by agegroup
# def plot(column):
#     if column == 'AgeGroup':
#         age_order = ['0-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71+']
#         df[column] = pd.Categorical(df[column], categories=age_order, ordered=True)

#     sns.histplot(x=column, hue="Exited", data=df, kde=True, palette="Set2")
#     plt.title(f"Distribution of {column} by Churn Status", fontweight="black", pad=20, size=15)
#     plt.figure(figsize=(13,6))
#     plt.show()

# print(plot('AgeGroup'))

##################
#  Feature Engineering
##################
## Products
# conditions = [(df["NumOfProducts"]==1), (df["NumOfProducts"]==2), (df["NumOfProducts"]>2)]
# values = ["One product","Two Products","More Than 2 Products"]
# df['Products'] = np.select(conditions,values)
#print(df.info())
#df.to_csv('C:/Users/Ha/Downloads/Churn_Modelling(1).csv', index=False)


######################
#  Data preprocessing 
######################
cols = ["Geography","Gender","Products","Acct Balance",'AgeGroup','Credit Score']
# for col in cols:
#      print(f"Unique values in {col} column is:",df[col].unique())
#      print('--------------------------\n')

# # One hot encoding on categorical cols
df_new = pd.get_dummies(columns = cols,data=df)
#print(df_new.info())

# # Encoding target variable
# df_new['Exited'].replace({'No':0,'Yes':1},inplace=True)
# #print(df_new['Exited'])

# # Checking skewness of continuous cols
# #check = df_new['EstimatedSalary'].skew()
# #print(check) # value: 0.0020853576615585162

# sns.histplot(df_new['EstimatedSalary'],kde=True)
# plt.title('Distribution of EstimatedSalary')
# plt.show()

# # Segregating features & Labels for model training
# X = df_new.drop(columns=['Exited'])
# y = df_new['Exited']

# # Splitting Data for model training & testing
# x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)
# # # print('Shape of x_train is: ',x_train.shape)
# # # print('Shape of x_test is: ',x_test.shape)
# # # print('Shape of y_train is: ',y_train.shape)
# # # print('Shape of y_test is: ',y_test.shape)

# # # Applying SMOTE to overcome the class imbalance in target variable
# smt = SMOTE(random_state=42)
# x_train_resampled,y_train_resampled = smt.fit_resample(x_train,y_train)
#print(x_train_resampled.shape,y_train_resampled.shape)
# ## print(y_train_resampled.value_counts().to_frame()) #both the categories in 'Exited' col are having equal number

# ###################################
#     # Model using Decision Tree
# ###################################

# # Performing Grid_Search to find the best parameter
# tree = DecisionTreeClassifier()
# param_grid = {"max_depth":[3,4,5,6,7,8,9,10],
#               "min_samples_split":[2,3,4,5,6,7,8],
#               "min_samples_leaf":[1,2,3,4,5,6,7,8],
#               "criterion":["gini"],
#               "splitter":["best","random"],
#               "max_features": [None, 'sqrt', 'log2'],
#               "random_state": [42]}
# grid_search = GridSearchCV(tree, param_grid, cv=5, n_jobs=-1,verbose=1)

# grid_search.fit(x_train_resampled,y_train_resampled)
# best_param = grid_search.best_params_
# #print('Best Parameters for DecisionTree Model: \n',best_param)

# # Creating DecisionTree model using best parameter
# tree = DecisionTreeClassifier(**best_param)
# tree.fit(x_train_resampled,y_train_resampled)

# # Computing model accuracy 
# y_train_predict = tree.predict(x_train_resampled)
# y_test_predict = tree.predict(x_test)

##print('Accuracy Score of Model on Training Data:',round(accuracy_score(y_train_resampled,y_train_predict)*100,2),'%')
# # Accuracy score of model: 90.44%
# #print('Accuracy Score of Model on Testing Data is',round(accuracy_score(y_test,y_test_predict)*100,2),'%')
# # Accuracy score of model: 84.85%

# # Model evaluation using different metric values
# # print('F1 Score of the model is: ',f1_score(y_test,y_test_predict,average='micro'))
# # print('Recall Score of the model is: ',recall_score(y_test,y_test_predict,average='micro'))
# # print('Precision Score of the model is: ',precision_score(y_test,y_test_predict,average='micro'))

# #Finding important factors in DecisionTree
# # factors = pd.DataFrame({'Feature Name':x_train.columns,'Importance':tree.feature_importances_})
# # features = factors.sort_values(by='Importance',ascending=False)
# # plt.figure(figsize=(12,7))
# # sns.barplot(x='Importance',y='Feature Name',data=features,palette='coolwarm')
# # plt.title('Important Factors in the model prediction',color='red',fontweight='bold',size=20,pad=20)
# # plt.yticks(size=12)
# # #plt.show()

# # Model evaluation using Confusion Matrix
# cm = confusion_matrix(y_test,y_test_predict)
# sns.heatmap(cm, 
#             annot=True,
#             fmt='g', 
#             xticklabels=['Not Exited','Exited'],
#             yticklabels=['Not Exited','Exited'])
# # Adding titles and labels
# plt.ylabel('Actual', fontsize=13)
# plt.xlabel('Prediction', fontsize=13)
# plt.title('Model Evaluation using Confusion Matrix', fontsize=17, pad=20)

# # Adjustments for better visualization
# plt.gca().xaxis.set_label_position('top') 
# plt.gca().xaxis.tick_top()
#plt.show()

############################################
    # Model creation using RandomForest
############################################

# # Find the best parameters for the model
# forest = RandomForestClassifier()
# param_grid = {"max_depth":[5,6,7,8],
#               "min_samples_split":[4,5],
#               "min_samples_leaf":[4,5],
#               "n_estimators": [70,100],
#               "criterion":["gini"]}
# grid_search = GridSearchCV(forest,param_grid,cv=5,n_jobs=-1)
# grid_search.fit(x_train_resampled,y_train_resampled)

# params = grid_search.best_params_
#print('Best parameters for model:\n',params)

#######
 # Creating model using best parameters
#######
# forest = RandomForestClassifier(**params)
# forest.fit(x_train_resampled,y_train_resampled)

# #######
#  # Testing model accuracy
# #######
# y_train_predict = forest.predict(x_train_resampled)
# y_test_predict = forest.predict(x_test)
#print("Score of Model by using Training Data: ",round(
    #accuracy_score(y_train_resampled,y_train_predict)*100,2),'%')
# Score: 90.26%
#print("Score of Model by using Testing Data: ",round(
    #accuracy_score(y_test,y_test_predict)*100,2),'%')
# Score: 84.4%

#######
 # Model evaluation
#######
#print("F1 score: ",f1_score(y_test,y_test_predict,average='micro'))
 # ==> 0.8435
#print('Recall Score: ',recall_score(y_test,y_test_predict,average='micro'))
 # ==> 0.8435
#print('Precision Score: ',precision_score(y_test,y_test_predict,average='micro'))
 # ==> 0.8435

#######
 # Finding factors affects the model
#######
# factors = pd.DataFrame({'Feature Name':x_train.columns,'Importance':forest.feature_importances_})
# features = factors.sort_values(by='Importance',ascending=False)
# plt.figure(figsize=(12,7))
# sns.barplot(x='Importance',y='Feature Name',data=features,palette='coolwarm')
# plt.title('Important Factors in the model prediction',color='red',fontweight='bold',size=20,pad=20)
# plt.yticks(size=12)
# #plt.show()

#######
 # Model evaluation using confusion matrix
#######
# cm = confusion_matrix(y_test,y_test_predict)
# sns.heatmap(cm, 
#              annot=True,
#              fmt='g', 
#              xticklabels=['Not Exited','Exited'],
#              yticklabels=['Not Exited','Exited'])
# # # Adding titles and labels
# plt.ylabel('Actual', fontsize=13)
# plt.xlabel('Prediction', fontsize=13)
# plt.title('Model Evaluation using Confusion Matrix', fontsize=17, pad=20)

# # # Adjustments for better visualization
# plt.gca().xaxis.set_label_position('top') 
# plt.gca().xaxis.tick_top()
# plt.show()