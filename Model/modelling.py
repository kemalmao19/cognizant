import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

print("Reading data ...")


# Read file
def goRead(name):
    df = pd.read_csv("../Model_Interpretations/" + name)
    return df


# Drop unnamed: 0
def dropUnnamed(data):
    data.drop(columns=["Unnamed: 0"], inplace=True, errors="ignore")
    return data


sales_df = dropUnnamed(goRead("sales.csv"))
stock_df = dropUnnamed(goRead("sensor_stock_levels.csv"))
storage_df = dropUnnamed(goRead("sensor_storage_temperature.csv"))

print("Preparing & cleaning data ...")


# set datetime format
def setTime(data):
    data["timestamp"] = pd.to_datetime(data["timestamp"], format="%Y-%m-%d %H:%M:%S")
    return data["timestamp"]


# get hourly data
def getHours(x):
    x = x.dt.floor("H")
    return x


sales_df["timestamp"] = getHours(setTime(sales_df))
stock_df["timestamp"] = getHours(setTime(stock_df))
storage_df["timestamp"] = getHours(setTime(storage_df))


# aggregating data
def aggTable(file, group: list, agg: dict):
    """Aggregating data"""
    file = file.groupby(group).agg(agg).reset_index()
    return file


group = ["timestamp", "product_id"]
agg_sales = {"quantity": "sum"}
agg_stock = {"estimated_stock_pct": "mean"}
agg_storage = {"temperature": "mean"}

sales_group = aggTable(sales_df, group, agg_sales)
stock_group = aggTable(stock_df, group, agg_stock)
storage_group = aggTable(storage_df, group[0], agg_storage)


# merge main Table
def mergeTable(bfr, aftr, on: list):
    merged = bfr.merge(aftr, on=on, how="left")
    return merged


merged_df = mergeTable(stock_group, sales_group, group)
merged_df = mergeTable(merged_df, storage_group, group[0])

product_categories = sales_df[
    ["product_id", "category", "unit_price", "customer_type", "total", "payment_type"]
]

product_categories = product_categories.drop_duplicates()

merged_df = mergeTable(merged_df, product_categories, group[1])

# fill Null values
merged_df["quantity"] = merged_df["quantity"].fillna(0)
# drop duplicates
merged_df = merged_df.drop_duplicates()

print('Modeling...')

# Feature engineering
merged_df.drop(columns=['product_id'], inplace=True)

y = merged_df['estimated_stock_pct']
X = merged_df.drop(['estimated_stock_pct', 'timestamp', 'date_only'], axis=1)

K = 10 #Many folds we want to complete in this training
split = 0.75 #split 75/25%

accuracy = []

for fold in range(0, K):

  # Instantiate algorithm
  model = RandomForestRegressor()
  scaler = StandardScaler()

  # Create training and test samples
  X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=split, random_state=42)

  # Scale X data, we scale the data because it helps the algorithm to converge
  # and helps the algorithm to not be greedy with large values
  scaler.fit(X_train)
  X_train = scaler.transform(X_train)
  X_test = scaler.transform(X_test)

  # Train model
  trained_model = model.fit(X_train, y_train)

  # Generate predictions on test sample
  y_pred = trained_model.predict(X_test)

  # Compute accuracy, using mean absolute error
  mae = mean_absolute_error(y_true=y_test, y_pred=y_pred)
  accuracy.append(mae)
  print(f"Fold {fold + 1}: MAE = {mae:.3f}")

print(f"Average MAE Model: {(sum(accuracy) / len(accuracy)):.2f}")
