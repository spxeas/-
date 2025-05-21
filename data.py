import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import os
import json
from datetime import datetime
# import time # 不再需要，因為資料產生已移出

print("Program starting...")

# --- 1. Load Data ---
file_name = 'pedestrian_crossing_data.csv'
print(f"--- Loading Data from {file_name} ---")
try:
    simulated_data = pd.read_csv(file_name)
    print("Data loaded successfully.")
    print("\nLoaded data (first 5 rows):")
    print(simulated_data.head())
except FileNotFoundError:
    print(f"Error: {file_name} not found. Please run generate_data.py first to create the dataset.")
    exit()
except Exception as e:
    print(f"An error occurred while loading the data: {e}")
    exit()

# --- 2. Prepare Machine Learning Data ---
# 只使用行人數量作為特徵
X = simulated_data[['Pedestrian Count']]
y = simulated_data['Required Green Light Time (s)']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)  # 不使用固定的random_state

print(f"\n--- Data Split ---")
print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Data variance: {y.var():.2f}")  # 顯示數據的方差

# --- 4. Train Model ---
print(f"\n--- Model Training ---")
model = LinearRegression()
model.fit(X_train, y_train)
print("Linear regression model training completed.")

# --- 5. Evaluate Model & Remove Outliers Iteratively ---
print(f"\n--- Model Evaluation & Outlier Removal ---")

# 初始模型評估
def evaluate_model(X, y, model):
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y, y_pred)
    return mse, rmse, r2

# 複製原始數據用於迭代處理
working_data = simulated_data.copy()
X_working = working_data[['Pedestrian Count']]
y_working = working_data['Required Green Light Time (s)']

# 初始模型
model = LinearRegression()

# 迭代移除離群點
num_iterations = 5  # 設定迭代次數
models = []  # 儲存每次迭代的模型
removed_points = []  # 儲存被移除的點

print("\nIterative outlier removal process:")
print("Iteration | Points | RMSE  | R² Score | Equation")
print("-" * 60)

for i in range(num_iterations + 1):
    # 訓練當前數據的模型
    model = LinearRegression()
    model.fit(X_working, y_working)
    
    # 評估模型
    _, rmse, r2 = evaluate_model(X_working, y_working, model)
    
    # 記錄當前模型
    models.append({
        'model': model,
        'data': working_data.copy(),
        'rmse': rmse,
        'r2': r2,
        'equation': f"y = {model.coef_[0]:.2f}x + {model.intercept_:.2f}"
    })
    
    # 列印此迭代的結果
    print(f"{i:9d} | {len(working_data):6d} | {rmse:.2f} | {r2:.2f} | {models[-1]['equation']}")
    
    # 最後一次迭代不需要再移除點
    if i == num_iterations:
        break
    
    # 計算每個點到回歸線的距離
    y_pred = model.predict(X_working)
    residuals = y_working - y_pred
    abs_residuals = np.abs(residuals)
    
    # 總是找出並移除具有最大絕對殘差的點
    max_residual_idx = abs_residuals.idxmax()
    max_residual_point = working_data.loc[max_residual_idx]
    removed_points.append(max_residual_point)
    print(f"    Removing point with max residual: #{max_residual_idx}")
    working_data = working_data.drop(max_residual_idx)
    
    X_working = working_data[['Pedestrian Count']]
    y_working = working_data['Required Green Light Time (s)']

# 選擇最終模型 (使用最後一次迭代的模型)
final_model = models[-1]['model']
final_data = models[-1]['data']
print("\nFinal model after removing outliers:")
print(f"Points remaining: {len(final_data)}")
print(f"Final equation: {models[-1]['equation']}")
print(f"RMSE: {models[-1]['rmse']:.2f}")
print(f"R² Score: {models[-1]['r2']:.2f}")

# 用於預測和視覺化的模型
model = final_model
simulated_data = final_data

# --- 7. Visualize Results with Multiple Regression Lines ---
print(f"\n--- Result Visualization ---")

# 1. 散點圖與多條回歸線
plt.figure(figsize=(12, 8))

# 繪製所有數據點（包括移除的）
original_data = pd.concat([final_data, pd.DataFrame(removed_points)])
plt.scatter(original_data['Pedestrian Count'], original_data['Required Green Light Time (s)'], 
           alpha=0.6, label='All Data Points', c='lightgray')

# 繪製保留的數據點
plt.scatter(final_data['Pedestrian Count'], final_data['Required Green Light Time (s)'], 
           alpha=0.7, label='Retained Data', c='blue')

# 繪製移除的數據點
for i, point in enumerate(removed_points):
    plt.scatter(point['Pedestrian Count'], point['Required Green Light Time (s)'], 
                c='red', s=100, edgecolor='black', 
                label='Removed Outliers' if i == 0 else "")

# 繪製多條回歸線
x_range = np.linspace(10, 40, 100)

colors = plt.cm.viridis(np.linspace(0, 1, len(models)))

for i, model_info in enumerate(models):
    # 只繪製初始回歸線和最終回歸線
    if i != 0 and i != len(models) - 1:
        continue

    model = model_info['model']
    y_pred = model.predict(pd.DataFrame({'Pedestrian Count': x_range}))
    
    if i == 0:
        line_label = 'Initial Regression'
        linestyle = '--'
        linewidth = 1.5
        plot_color = colors[0]  # 使用顏色列表的第一個顏色
    elif i == len(models) - 1:
        line_label = 'Final Regression'
        linestyle = '-'
        linewidth = 2.5
        plot_color = colors[-1]  # 使用顏色列表的最後一個顏色
    else:
        # 這部分理論上不會執行到，但保留以防萬一
        line_label = f'Iteration {i}'
        linestyle = ':'
        linewidth = 1.5
        plot_color = colors[i]
    
    plt.plot(x_range, y_pred, linestyle=linestyle, linewidth=linewidth,
             color=plot_color, label=line_label)

plt.xlabel('Pedestrian Count')
plt.ylabel('Required Green Light Time (s)')
plt.title('Pedestrian Count vs. Green Light Time (Outlier Removal Process)')
plt.grid(True)
plt.legend()

# 在圖上顯示最終回歸方程式
equation = f"Final: {models[-1]['equation']}"
plt.text(0.05, 0.95, equation, transform=plt.gca().transAxes, 
         bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('pedestrian_crossing_regression_iterations.png')

# 2. 箱型圖 (使用最終數據)
plt.figure(figsize=(10, 6))
pedestrian_categories = sorted(final_data['Pedestrian Count'].unique())
# 過濾掉沒有數據的類別
filtered_categories = []
filtered_data = []
for count in pedestrian_categories:
    data = final_data[final_data['Pedestrian Count'] == count]['Required Green Light Time (s)']
    if len(data) > 0:
        filtered_categories.append(count)
        filtered_data.append(data)

plt.boxplot(filtered_data, labels=filtered_categories)
plt.xlabel('Pedestrian Count')
plt.ylabel('Required Green Light Time (s)')
plt.title('Distribution of Green Light Time by Pedestrian Count (After Outlier Removal)')
plt.grid(True)
plt.savefig('pedestrian_crossing_boxplot_final.png')

# plt.show() # 註解掉此行，使圖片不顯示

# --- 8. Save Updated Data ---
print(f"\n--- Saving Updated Data to {file_name} ---")
try:
    final_data.to_csv(file_name, index=False, encoding='utf-8')
    print(f"Updated data saved successfully to {file_name}.")
    print(f"Number of data points remaining: {len(final_data)}")
except Exception as e:
    print(f"An error occurred while saving the updated data: {e}")

print("Program execution completed. Charts saved as:")
print("- pedestrian_crossing_regression_iterations.png (Regression lines evolution)")
print("- pedestrian_crossing_boxplot_final.png (Box plot with final data)")

print("\nTo predict green light time, use the final formula:")
print(f"Green Light Time = {final_model.intercept_:.2f} + {final_model.coef_[0]:.2f} × Pedestrian Count")