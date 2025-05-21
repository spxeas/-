import numpy as np
import pandas as pd
import time

# --- Data Generation Function ---
def generate_pedestrian_crossing_data(
    num_samples: int = 150,
    min_pedestrians: int = 10,
    max_pedestrians: int = 40,
    road_width: float = 10.0,  # 固定路寬為10公尺
    walking_speed: float = 1.2,  # 固定步行速率
    reaction_time: float = 2.0,
    startup_time: float = 3.0,
    safety_margin: float = 3.0,
    noise_level: float = 3.0,  # 增加雜訊水平
    random_seed = None  # 使用隨機種子
):
    # 設定隨機種子
    if random_seed is not None:
        np.random.seed(random_seed)
    else:
        # 使用當前時間作為種子
        np.random.seed(int(time.time()))
    
    # 生成均勻分布的行人數量，確保每個數值都有樣本
    pedestrian_range = np.arange(min_pedestrians, max_pedestrians + 1)
    num_unique = len(pedestrian_range)
    samples_per_count = num_samples // num_unique
    pedestrian_counts = np.repeat(pedestrian_range, samples_per_count)
    
    # 如果樣本數不是行人數量範圍的整數倍，補足剩餘樣本
    if len(pedestrian_counts) < num_samples:
        extra_samples = np.random.choice(pedestrian_range, num_samples - len(pedestrian_counts))
        pedestrian_counts = np.concatenate([pedestrian_counts, extra_samples])
    
    # 打亂順序
    np.random.shuffle(pedestrian_counts)
    
    # 使用固定路寬但添加微小變化
    road_widths = np.full(num_samples, road_width) + np.random.normal(0, 0.1, num_samples)
    
    # 步行速率加入變化
    walking_speeds = np.random.normal(walking_speed, 0.15, num_samples)
    walking_speeds = np.clip(walking_speeds, 0.9, 1.5)  # 限制在合理範圍內
    
    # 計算基本時間，加入變異
    base_times = reaction_time + startup_time + (road_widths / walking_speeds) + safety_margin
    base_times = base_times + np.random.normal(0, 1.0, num_samples)  # 基本時間加入隨機變異
    
    # 行人數量對時間的影響，非完美線性
    coef_variance = np.random.normal(0.5, 0.1, num_samples)  # 係數本身有變化
    pedestrian_effect = pedestrian_counts * coef_variance
    
    # 理論時間
    theoretical_time = base_times + pedestrian_effect
    
    # 加入雜訊
    noise = np.random.normal(0, noise_level, num_samples)
    required_green_time = theoretical_time + noise
    required_green_time = np.clip(required_green_time, 10.0, 60.0)
    
    data = pd.DataFrame({
        'Pedestrian Count': pedestrian_counts,
        'Road Width (m)': road_widths.round(1),
        'Walking Speed (m/s)': walking_speeds.round(2),
        'Required Green Light Time (s)': required_green_time.round(1)
    })
    return data

if __name__ == "__main__":
    print("Generating pedestrian crossing data...")
    
    # 使用當前時間作為種子
    current_seed = int(time.time()) % 10000  # 取模，使種子保持在合理範圍內
    print(f"Using seed: {current_seed}")
    
    simulated_data = generate_pedestrian_crossing_data(random_seed=current_seed)
    
    file_name = 'pedestrian_crossing_data.csv'
    try:
        simulated_data.to_csv(file_name, index=False, encoding='utf-8')
        print(f"Data saved as {file_name}")
        print(f"Generated {len(simulated_data)} samples.")
        print("\nFirst 5 rows of generated data:")
        print(simulated_data.head())
    except Exception as e:
        print(f"Error occurred while saving file: {e}")
    
    print("Data generation complete.") 