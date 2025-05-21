import subprocess
import json
import os

NUM_ITERATIONS = 20


# 首先執行 generate_data.py 一次
print("Running generate_data.py once...")
process_generate = subprocess.run(["python", "generate_data.py"], capture_output=True, text=True)
if process_generate.returncode != 0:
    print("Error running generate_data.py:")
    print(process_generate.stderr)
    exit()  # 如果 generate_data.py 出錯則停止
print(process_generate.stdout)
print("generate_data.py finished.")

print("\nStarting iterations for data.py...")
# 然後執行 data.py 50 次
for i in range(NUM_ITERATIONS):
    print(f"Iteration {i+1}/{NUM_ITERATIONS} for data.py")

    # 執行 data.py
    print("Running data.py...")
    process_data = subprocess.run(["python", "data.py"], capture_output=True, text=True)
    if process_data.returncode != 0:
        print(f"Error running data.py on iteration {i+1}:")
        print(process_data.stderr)
        break  # 如果 data.py 出錯則停止
    print(process_data.stdout)
    print("data.py finished.")

print("All data.py iterations completed.")

# 將最後一次 data.py 的結果附加到 final_model_results.json
latest_result_file = '_latest_data_run_output.json'
final_results_file = 'final_model_results.json'

if os.path.exists(latest_result_file):
    try:
        with open(latest_result_file, 'r', encoding='utf-8') as f_latest:
            latest_run_data = json.load(f_latest)

        all_results = []
        if os.path.exists(final_results_file):
            try:
                with open(final_results_file, 'r', encoding='utf-8') as f_final:
                    # 檢查檔案是否為空或內容無效
                    content = f_final.read()
                    if content.strip(): # 如果檔案不為空
                        f_final.seek(0) # 重置讀取位置
                        all_results = json.load(f_final)
                        if not isinstance(all_results, list):
                            print(f"Warning: {final_results_file} does not contain a list. It will be overwritten.")
                            all_results = [] # 如果不是列表，則重置
                    else:
                        all_results = [] # 如果檔案為空，則初始化為空列表
            except json.JSONDecodeError:
                print(f"Warning: Could not decode JSON from {final_results_file}. It will be overwritten.")
                all_results = []
            except Exception as e:
                print(f"Error reading {final_results_file}: {e}. It might be overwritten.")
                all_results = [] # 其他錯誤也重置
        
        all_results.append(latest_run_data)

        with open(final_results_file, 'w', encoding='utf-8') as f_final_out:
            json.dump(all_results, f_final_out, ensure_ascii=False, indent=4)
        print(f"Appended latest run results to {final_results_file}")

    except Exception as e:
        print(f"Error processing result files: {e}")
else:
    print(f"Warning: {latest_result_file} not found. Cannot append results.")

print("run_scripts.py finished.")
