import matplotlib.pyplot as plt

# Data
parameters = [2, 4, 8, 16]
psnr = [27.608, 28.148, 28.304, 27.798]
fid = [69.626, 61.321, 76.401, 89.157]
lpips = [0.183, 0.175, 0.194, 0.225]
runtime = [0.164, 0.208, 0.295, 0.471]

# Initialize plot
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
fig, ax1 = plt.subplots(figsize=(10, 6))

# Left Y-axis (PSNR and Quality Metrics)
color_psnr = '#1f77b4'
ax1.set_xlabel('Parameter', fontsize=14)
ax1.set_ylabel('PSNR (dB)', color=color_psnr, fontsize=14)
ax1.plot(parameters, psnr, color=color_psnr, marker='o', markersize=8, linewidth=2, label='PSNR')
ax1.tick_params(axis='y', labelcolor=color_psnr)
ax1.set_ylim(27, 28.5)
ax1.set_xticks(parameters)

# Right Y-axis (Runtime)
ax2 = ax1.twinx()
color_runtime = '#d62728'
ax2.set_ylabel('Runtime (s)', color=color_runtime, fontsize=14)
ax2.plot(parameters, runtime, color=color_runtime, marker='s', markersize=8, linewidth=2, label='Runtime')
ax2.tick_params(axis='y', labelcolor=color_runtime)
ax2.set_ylim(0.1, 0.5)

# Add text annotations for FID and LPIPS
for x, p, f, l in zip(parameters, psnr, fid, lpips):
    ax1.text(x, p-0.15, f'FID: {f}\nLPIPS: {l:.3f}', 
            ha='center', va='top', fontsize=10, 
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

# Legend
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc='upper left', fontsize=12)

plt.title('Performance Metrics Comparison', fontsize=16, pad=20)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()