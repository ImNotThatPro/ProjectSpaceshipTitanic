import pandas as pd
import random
import numpy as np
from PIL.PcfFontFile import BYTES_PER_ROW
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale, MinMaxScaler

df_test = pd.read_csv('test.csv')
df_train = pd.read_csv('train.csv')

#cleaning df_train
#
#
#Dropping useless columns
df_train = df_train.drop(columns = ['Destination', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck','Name'])
#Split ['Cabin'] column into three readable values columns(this is gonna help the model calculate more precise?)
df_train[['Deck', 'CabinNum', 'Side']] = df_train['Cabin'].str.split('/', expand = True)
df_train = df_train.drop(columns = ['Cabin'])

df_train['HomePlanet'] = df_train['HomePlanet'].fillna('Earth')
df_train['CryoSleep'] = df_train['CryoSleep'].fillna('False')
#Using mean method to calculate the average age of the age column then fill the NaN value inside the Age column
mean_age = df_train['Age'].mean()
df_train['Age'] = df_train['Age'].fillna(mean_age)
df_train['VIP'] = df_train['VIP'].fillna('False')
#Some how this work and i don't understand how did this work
df_train['Side'] = df_train['Side'].apply(
    lambda value: np.random.choice(['S', 'P']) if pd.isna(value) else value
)
df_train['Deck'] = df_train['Deck'].apply(
    lambda value: np.random.choice(['B','C','D','E','F','G']) if pd.isna(value) else value
)
df_train['CabinNum'] = df_train['CabinNum'].fillna(0)
print(df_train.isnull().sum())
print(df_train.sample)
#Training model
#
#
#Splitting data into two
X = df_train.drop(columns=['Transported'])
y = df_train['Transported']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25 )

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_prediction = model.predict(X_test)
print(f'Model accuracy: {accuracy_score(y_test, y_prediction)}')