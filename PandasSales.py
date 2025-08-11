import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', None)
#Cleaning data
sales_data = pd.read_csv('C:/Users/duong/PycharmProjects/PythonProject1/PandasLearningProjects/fake_sales_data.csv')

sales_data['Revenue'] = sales_data['Quantity Ordered'] * sales_data['Price Each']
product_totals = sales_data.groupby('Product')['Quantity Ordered'].sum()

revenue_totals = sales_data.groupby('Product')['Revenue'].sum()
sales_data['Most Sold Products'] = sales_data['Product'].map(revenue_totals)

sales_data['Order Date'] = pd.to_datetime(sales_data['Order Date'], errors = 'coerce')
sales_data['Month'] = sales_data['Order Date'].dt.to_period('M')

revenue_month = sales_data.groupby('Month')['Revenue'].sum()
sales_data['Revenue Month'] = sales_data['Month'].map(revenue_month)

new_columns_order = ['Order ID', 'Product', 'Quantity Ordered', 'Price Each', 'Revenue', 'Most Sold Products',
                     'Order Date','Month', 'Revenue Month', 'Purchase Address']

sales_data = sales_data[new_columns_order]
print(sales_data.head().to_string(index= False))


plt.figure(figsize=(10,6 ))
plt.bar(revenue_month.index.astype(str), revenue_month.values, color = 'skyblue')
plt.title('Total Revenue Per Month')
plt.xlabel('Month')
plt.ylabel('Revenue $')
plt.xticks(rotation = 45)
plt.tight_layout()
plt.show()