import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
suffix = 'lambda_0_10_1'

accuracy_data = np.loadtxt(f"nn_training_accuracy_record_{suffix}.txt", delimiter=',', dtype=[('n_hidden', 'i4'), ('lambdaval', 'f4'), ('accuracy', 'f4'), ('training_accuracy', 'f4')])
time_data = np.loadtxt(f"nn_training_time_record_{suffix}.txt", delimiter=',', dtype=[('n_hidden', 'i4'), ('lambdaval', 'f4'), ('accuracy', 'f4')])

# # Convert data of accuracy_data
# n_hidden = [data[0] for data in accuracy_data]
# lambdaval = [data[1] for data in accuracy_data]
# accuracy = [data[2] for data in accuracy_data]
# accuracy_heatmap = pd.DataFrame({
#     'n_hidden': n_hidden,
#     'lambdaval': lambdaval,
#     'accuracy': accuracy
# })
# accuracy_heatmap_pivot = accuracy_heatmap.pivot(index="n_hidden", columns="lambdaval", values="accuracy")

# # Plot the accuracy heatmap
# plt.figure(figsize=(12, 8))
# sns.heatmap(accuracy_heatmap_pivot, annot=True, cmap="YlGnBu", fmt=".2f", cbar=True)
# plt.title("Heatmap of Validation Accuracies")
# plt.ylabel("Number of Hidden Units (n_hidden)")
# plt.xlabel("Lambda (Regularization Parameter)")
# plt.savefig("accuracy_heatmap_smallrange.png")






# # Convert data of time_data
# n_hidden = [data[0] for data in time_data if data[1] == 0]
# # lambdaval = [data[1] for data in time_data]
# times = [data[2] for data in time_data if data[1] == 0]
# plt.figure(figsize=(12, 8))

# # Plot with enhancements
# plt.plot(n_hidden, times, marker = 'o', linestyle = '--', color = 'royalblue', markersize = 8, linewidth = 2)

# # Adding point annotations
# for i, (x, y) in enumerate(zip(n_hidden, times)):
#     plt.text(x, y + 0.5, f"{y:.2f}s", ha='center', fontsize = 10, color = 'darkblue')

# plt.title("Training Time with Different Hidden Units (n_hidden)", fontsize = 16, fontweight = 'bold')
# plt.xlabel("Number of Hidden Units (n_hidden)", fontsize=14, fontweight ='bold')
# plt.ylabel("Training Time (s)", fontsize = 14, fontweight='bold')
# plt.grid(visible = True, which = 'both', color = 'gray', linestyle = '--', linewidth = 0.5)
# plt.fill_between(n_hidden, times, color='lightblue', alpha=0.3)
# plt.savefig("time_plot_smallrange.png")





# Convert data of time_data
# n_hidden = [data[0] for data in time_data if data[1] == 0]
lambdaval = [data[1] for data in accuracy_data]
accuracys = [data[2] for data in accuracy_data]
training_acc = [data[3] for data in accuracy_data]
# times = [data[2] for data in time_data if data[1] == 0]
plt.figure()

# Plot with enhancements
plt.plot(lambdaval, accuracys, marker = 'o', linestyle = '--', color = 'royalblue', markersize = 8, linewidth = 2)
plt.plot(lambdaval, training_acc, marker = 'o', linestyle = '--', color = 'orange', markersize = 8, linewidth = 2)

# # Adding point annotations
# for i, (x, y) in enumerate(zip(lambdaval, accuracys)):
#     plt.text(x, y + 0.5, f"{y:.2f}s", ha='center', fontsize = 10, color = 'darkblue')

plt.grid(True) #Adding grid for better readability
plt.xlim(0, 10) #Setting x-axis limits
plt.ylim(95 - 5 - 0.1, 95 + 5 + 0.1) #Setting y-axis limits to focus on the oscillations


plt.title("Acc with Different Lambda", fontsize = 16, fontweight = 'bold')
plt.xlabel("Lambda", fontsize=14, fontweight ='bold')
plt.ylabel("Acc", fontsize = 14, fontweight='bold')
plt.grid(visible = True, which = 'both', color = 'gray', linestyle = '--', linewidth = 0.5)
plt.fill_between(lambdaval, accuracys, color='lightblue', alpha=0.3)
plt.savefig(f"acc_lambda_{suffix}.png")