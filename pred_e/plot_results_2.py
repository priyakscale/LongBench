import json
import os
import matplotlib.pyplot as plt
import numpy as np

# Load the JSON data
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

lines = []
labels = []

for idx, dataset in enumerate(datasets):
    ax = axes[idx]
    
    for i, file_data in enumerate(data):
        performance = [file_data[dataset].get(r, 0) for r in ranges]
        label = None
        if i == 0:
            label = 'llama3 base 8k'
        elif i == 1:
            label = 'llama3 base + smart compression 8k'
        elif i == 2:
            label = 'llama3 ift 4k'
        elif i == 3:
            label = 'llama3 ift + smart compression 4k'
        elif i == 4:
            label = 'llama3 instruct rope-scale linear 16k'
        elif i == 5:
            label = 'llama3 instruct rope-scale ntk dynamic 16k'
        elif i == 6:
            label = 'gpt-4o base 128k'
        
        line, = ax.plot(ranges, performance, marker='o')
        if label not in labels:
            lines.append(line)
            labels.append(label)
    
    ax.set_title(f'Performance for {dataset}')
    ax.set_xlabel('Range')
    ax.set_ylabel('Performance')
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
    performance = [average_performance[r] for r in ranges]
    label = None
    if i == 0:
        label = 'llama3 base 8k'
    elif i == 1:
        label = 'llama3 base + smart compression 8k'
    elif i == 2:
        label = 'llama3 ift 4k'
    elif i == 3:
        label = 'llama3 ift + smart compression 4k'
    elif i == 4:
        label = 'llama3 instruct rope-scale linear 16k'
    elif i == 5:
        label = 'llama3 instruct rope-scale ntk dynamic 16k'
    elif i == 6:
        label = 'gpt-4o base 128k'
    
    line, = ax.plot(ranges, performance, marker='o')
    if label not in labels:
        lines.append(line)
        labels.append(label)

ax.set_title('Average Performance Across All Datasets')
ax.set_xlabel('Range')
ax.set_ylabel('Performance')
ax.grid(True)

# Add an empty subplot for the legend
fig.add_subplot(5, 3, 15)

# Create a combined legend
fig.legend(lines, labels, loc='center', bbox_to_anchor=(0.9, 0.1), ncol=1)

plt.tight_layout()
plt.savefig('combined_performance_grid_all_techniques.png')
plt.close()
#plt.show()
