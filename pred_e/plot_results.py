import json
import os
import matplotlib.pyplot as plt
import numpy as np

# Load the JSON data
# file_paths = [
#     '/mnt/efs/priyakhandelwal/src/LongBench/pred_e/llama3-8b-8k/result.json',
#     '/mnt/efs/priyakhandelwal/src/LongBench/pred_e/llama3-8b-8k+LLMLingua2/result.json',
#     '/mnt/efs/priyakhandelwal/src/LongBench/pred_e/llama3-8b-8k+LLMLingua2_relaxed/result.json'
# ]
# file_paths = [
#     '/mnt/efs/priyakhandelwal/src/LongBench/pred_e/llama3-8b-ift-2e-5/result.json',
#     '/mnt/efs/priyakhandelwal/src/LongBench/pred_e/llama3-8b-ift-2e-5+LLMLingua2/result.json',
#     '/mnt/efs/priyakhandelwal/src/LongBench/pred_e/llama3-8b-ift-5e-6/result.json',
#     '/mnt/efs/priyakhandelwal/src/LongBench/pred_e/llama3-8b-ift-5e-6+LLMLingua2/result.json',
#     '/mnt/efs/priyakhandelwal/src/LongBench/pred_e/llama3-8b-8k/result.json',
# ]
file_paths = [
    '/mnt/efs/priyakhandelwal/src/LongBench/pred_e/llama3-8b-8k/result.json',
    '/mnt/efs/priyakhandelwal/src/LongBench/pred_e/llama3-8b-8k+LLMLingua2/result.json',
    '/mnt/efs/priyakhandelwal/src/LongBench/pred_e/llama3-8b-ift-2e-5/result.json',
    '/mnt/efs/priyakhandelwal/src/LongBench/pred_e/llama3-8b-ift-2e-5+LLMLingua2/result.json',
    '/mnt/efs/priyakhandelwal/src/LongBench/pred_e/llama3_8b_instruct_ift_2e-5_dynamic/result.json',
    '/mnt/efs/priyakhandelwal/src/LongBench/pred_e/llama3_8b_instruct_ift_2e-5_linear/result.json',
    '/mnt/efs/priyakhandelwal/src/LongBench/pred_e/gpt-4o/result.json',
]

data = []
for file_path in file_paths:
    with open(file_path) as f:
        data.append(json.load(f))

# Extract the datasets and ranges
datasets = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news", \
            "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]
ranges = ["0-4k", "4-8k", "8k+"]

# Create a figure with a grid of subplots
fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(20, 20))

# Flatten the axes array for easier indexing
axes = axes.flatten()

for idx, dataset in enumerate(datasets):
    ax = axes[idx]
    
    for i, file_data in enumerate(data):
        performance = [file_data[dataset].get(r, 0) for r in ranges]
        if i == 0:
            ax.plot(ranges, performance, marker='o', label='llama3 base 8k')
        if i == 1:
            ax.plot(ranges, performance, marker='o', label='llama3 base + smart compression 8k')
        if i == 2:
            ax.plot(ranges, performance, marker='o', label='llama3 ift 4k')
        if i == 3:
            ax.plot(ranges, performance, marker='o', label='llama3 ift + smart compression 4k')
        if i == 4:
            ax.plot(ranges, performance, marker='o', label='llama3 instruct rope-scale linear 16k')
        if i == 5:
            ax.plot(ranges, performance, marker='o', label='llama3 instruct rope-scale ntk dynamic 16k')
        if i == 6:
            ax.plot(ranges, performance, marker='o', label='gpt-4o base 128k')
    
    ax.set_title(f'Performance for {dataset}')
    ax.set_xlabel('Range')
    ax.set_ylabel('Performance')
    ax.legend()
    ax.grid(True)

# Calculate and plot average performance across all datasets for each file
ax = axes[len(datasets)]  # Use the next subplot for the average performance plot

for i, file_data in enumerate(data):
    average_performance = {r: [] for r in ranges}  # Initialize average_performance as a dictionary of lists
    for dataset in datasets:
        performance = [file_data[dataset].get(r, 0) for r in ranges]
        for r, value in zip(ranges, performance):
            average_performance[r].append(value)
    
    average_performance = {r: np.mean(values) for r, values in average_performance.items()}
    if i == 0:
            ax.plot(ranges, performance, marker='o', label='llama3 base 8k')
    if i == 1:
        ax.plot(ranges, performance, marker='o', label='llama3 base + smart compression 8k')
    if i == 2:
        ax.plot(ranges, performance, marker='o', label='llama3 ift 4k')
    if i == 3:
        ax.plot(ranges, performance, marker='o', label='llama3 ift + smart compression 4k')
    if i == 4:
        ax.plot(ranges, performance, marker='o', label='llama3 instruct rope-scale linear 16k')
    if i == 5:
        ax.plot(ranges, performance, marker='o', label='llama3 instruct rope-scale ntk dynamic 16k')
    if i == 6:
        ax.plot(ranges, performance, marker='o', label='gpt-4o base 128k')

ax.set_title('Average Performance Across All Datasets')
ax.set_xlabel('Range')
ax.set_ylabel('Performance')
ax.legend()
ax.grid(True)

# Hide any unused subplots
for idx in range(len(datasets) + 1, len(axes)):
    fig.delaxes(axes[idx])

plt.tight_layout()
plt.savefig('combined_performance_grid_multiple_techniques.png')
plt.close()

# import json
# import os
# import matplotlib.pyplot as plt
# import numpy as np

# # Load the JSON data
# file_paths = [
#     '/mnt/efs/priyakhandelwal/src/LongBench/pred_e/llama3-8b-8k/result.json',
#     '/mnt/efs/priyakhandelwal/src/LongBench/pred_e/llama3-8b-8k+LLMLingua2/result.json',
#     '/mnt/efs/priyakhandelwal/src/LongBench/pred_e/llama3-8b-8k+LLMLingua2_relaxed/result.json'
# ]

# data = []
# for file_path in file_paths:
#     with open(file_path) as f:
#         data.append(json.load(f))

# # Extract the datasets and ranges
# datasets = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news", \
#             "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]
# ranges = ["0-4k", "4-8k", "8k+"]

# # Create a figure with multiple subplots
# num_datasets = len(datasets)
# fig, axes = plt.subplots(nrows=num_datasets, ncols=3, figsize=(10, num_datasets * 5))

# for idx, dataset in enumerate(datasets):
#     ax = axes[idx]
    
#     for i, file_data in enumerate(data):
#         performance = [file_data[dataset].get(r, 0) for r in ranges]
#         if i == 0:
#             ax.plot(ranges, performance, marker='o', label=f'Compress middle')
#         if i == 1:
#             ax.plot(ranges, performance, marker='o', label=f'Compress rate 30')
#         if i == 2:
#             ax.plot(ranges, performance, marker='o', label=f'Compress end')
    
#     ax.set_title(f'Performance for {dataset}')
#     ax.set_xlabel('Range')
#     ax.set_ylabel('Performance')
#     ax.legend()
#     ax.grid(True)

# plt.tight_layout()
# plt.savefig('combined_performance.png')
# plt.close()

# # Calculate and plot average performance across all datasets for each file
# plt.figure(figsize=(10, 6))
# for i, file_data in enumerate(data):
#     average_performance = {r: [] for r in ranges}
    
#     for dataset in datasets:
#         performance = [file_data[dataset].get(r, 0) for r in ranges]
#         for r, value in zip(ranges, performance):
#             average_performance[r].append(value)
    
#     average_performance = {r: np.mean(values) for r, values in average_performance.items()}
#     if i == 0:
#         plt.plot(ranges, list(average_performance.values()), marker='o', label=f'Compress middle')
#     if i == 1:
#         plt.plot(ranges, list(average_performance.values()), marker='o', label=f'Compress rate 30')
#     if i == 2:
#         plt.plot(ranges, list(average_performance.values()), marker='o', label=f'Compress end')

# plt.title('Average Performance Across All Datasets')
# plt.xlabel('Range')
# plt.ylabel('Performance')
# plt.legend()
# plt.grid(True)

# # Save the average performance plot
# plt.savefig('average_performance.png')
# plt.close()


# import json
# import os
# import matplotlib.pyplot as plt
# import numpy as np

# # Load the JSON data
# file_paths = [
#     '/mnt/efs/priyakhandelwal/src/LongBench/pred_e/llama3-8b-8k/result.json',
#     '/mnt/efs/priyakhandelwal/src/LongBench/pred_e/llama3-8b-8k+LLMLingua2/result.json',
#     '/mnt/efs/priyakhandelwal/src/LongBench/pred_e/llama3-8b-8k+LLMLingua2_relaxed/result.json'
# ]

# data = []
# for file_path in file_paths:
#     with open(file_path) as f:
#         data.append(json.load(f))

# # Extract the datasets and ranges
# datasets = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news", \
#             "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]
# ranges = ["0-4k", "4-8k", "8k+"]

# # Plot the performance for each dataset
# for dataset in datasets:
#     plt.figure(figsize=(10, 6))
    
#     for i, file_data in enumerate(data):
#         performance = [file_data[dataset].get(r, 0) for r in ranges]
#         if i == 0:
#             plt.plot(ranges, performance, marker='o', label=f'Compress middle')
#         if i == 1:
#             plt.plot(ranges, performance, marker='o', label=f'Compress rate 30')
#         if i == 2:
#             plt.plot(ranges, performance, marker='o', label=f'Compress end')
    
#     plt.title(f'Performance for {dataset}')
#     plt.xlabel('Range')
#     plt.ylabel('Performance')
#     plt.legend()
#     plt.grid(True)
    
#     # Save the plot
#     plt.savefig(f'{dataset}_performance.png')
#     plt.close()

# # Calculate and plot average performance across all datasets for each file
# plt.figure(figsize=(10, 6))
# for i, file_data in enumerate(data):
#     average_performance = {r: [] for r in ranges}
    
#     for dataset in datasets:
#         performance = [file_data[dataset].get(r, 0) for r in ranges]
#         for r, value in zip(ranges, performance):
#             average_performance[r].append(value)
    
#     average_performance = {r: np.mean(values) for r, values in average_performance.items()}
#     if i == 0:
#         plt.plot(ranges, list(average_performance.values()), marker='o', label=f'Compress middle')
#     if i == 1:
#         plt.plot(ranges, list(average_performance.values()), marker='o', label=f'Compress rate 30')
#     if i == 2:
#         plt.plot(ranges, list(average_performance.values()), marker='o', label=f'Compress end')

# plt.title('Average Performance Across All Datasets')
# plt.xlabel('Range')
# plt.ylabel('Performance')
# plt.legend()
# plt.grid(True)

# # Save the average performance plot
# plt.savefig('average_performance.png')
# plt.close()