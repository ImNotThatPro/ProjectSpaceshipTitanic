import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  MinMaxScaler, LabelEncoder

df_test = pd.read_csv('test.csv')
df_train = pd.read_csv('train.csv')

#cleaning df_train
#
#
#Cleaning function for both of the two dataframe
def clean_data(df):
    df = df.drop(columns = ['Destination', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck','Name'] )
    df[['Deck', 'CabinNum', 'Side']] = df['Cabin'].str.split('/', expand = True)
    df = df.drop(columns= ['Cabin'])
    df['HomePlanet'] = df['HomePlanet'].fillna('Earth')
    df['CryoSleep'] = df['CryoSleep'].fillna('False')
    mean_age = df['Age'].mean()
    df['Age'] = df['Age'].fillna(mean_age)
    df['VIP'] = df['VIP'].fillna(False)
    np.random.seed(42)
    df['Side'] = df['Side'].apply(
        lambda value: np.random.choice(['S','P']) if pd.isna(value) else value
    )
    df['Deck'] = df['Deck'].apply(
        lambda value: np.random.choice(['B','C','D','E','F','G']) if pd.isna(value) else value
    )
    df['CabinNum'] = df['CabinNum'].fillna(0)
    return df

df_train = clean_data(df_train)
df_test = clean_data(df_test)
print(df_train.isnull().sum())
print(df_train.info())
print(df_test.isnull().sum())
print(df_test.info())

#Training model
#
#
#Encoding non-numeric columns from data set
encoder = LabelEncoder()
for col in ['HomePlanet', 'CryoSleep', 'VIP','Deck','Side']:
    df_train[col] = encoder.transform(df_train[col])
print(df_train.info())

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