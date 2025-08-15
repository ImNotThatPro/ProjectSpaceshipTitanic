import pandas as pd
import matplotlib.pyplot as plt

hpdf = pd.read_csv('C:/Users/duong/PycharmProjects/PythonProject1/PandasLearningProjects/hospital_visit_system.csv')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
#imagine that we found out about our mistake and he was discharged on the 20 just to make it simple
#goal n1,2,3,8,9, ✅
hpdf.fillna({'Cost': hpdf['Cost'].mean()}, inplace=True)
hpdf.loc[hpdf['Name'] == 'David Clark', 'DischargeDate'] = '2025-04-20'
hpdf['DischargeDate'] = hpdf['DischargeDate'].astype(str)
#goal n4 ✅
gender_count = hpdf['Gender'].value_counts()
#goal n5 ✅
departments = hpdf['Department'].unique()
#goal n6✅
hpdf['AdmissionDate'] = pd.to_datetime(hpdf['AdmissionDate'], errors='coerce')
hpdf['DischargeDate'] = pd.to_datetime(hpdf['DischargeDate'], errors='coerce')
#goal n7✅
hpdf['Stayed Days'] = hpdf['DischargeDate'] - hpdf['AdmissionDate']
hpdf['Stayed Days'] = hpdf['Stayed Days'].dt.days
#goal n10✅
hpdf['Stayed Days'] = pd.to_numeric(hpdf['Stayed Days'], errors='coerce')
hpdf['Cost'] = hpdf['Cost'].round(2)
hpdf['Cost Per Day'] = hpdf['Cost'] / hpdf['Stayed Days']
hpdf['Cost Per Day'] = hpdf['Cost Per Day'].round(2)
#goal n11✅
average_stay_days = hpdf['Stayed Days'].mean()
#goal n12✅
department_revenue = hpdf.groupby('Department')['Cost'].sum()
#goal n13✅
need_follow_up = hpdf[hpdf['FollowUpRequired']== 'Yes'].groupby('Department').size()
#goal n14✅
most_popular_diagnosis = hpdf['Diagnosis'].value_counts()
# goal n15✅
oldest = hpdf.sort_values(by='Age', ascending=False).iloc[0]['Name']
#goal n16
patients_per_day = hpdf.sort_values(by='AdmissionDate', ascending = False)

plt.figure(figsize=(10,6))



columns_order = ['PatientID', 'Name', 'Age', 'Gender', 'Department',
                 'AdmissionDate', 'DischargeDate', 'Diagnosis',
                 'Stayed Days','Cost', 'Cost Per Day', 'FollowUpRequired']
hpdf = hpdf[columns_order]
print()
