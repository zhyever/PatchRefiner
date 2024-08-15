
import matplotlib.pyplot as plt



baseline_sqrel = 1.0256434
baseline_rmse = 8.3237575
baseline_edgecomp = 22.1390643
# baseline_edgecomp = 3.4841872
baseline_edgeacc = 3.3948909
baseline_f1 = 0.2699457

# ranking
# 0.0001 0.0003 0.0005 0.005
ranking_rmse = [8.3086033, 8.3105562, 8.3118163, 8.3268201]
ranking_edgecomp = [17.7865264, 17.7455107, 17.7040826, 17.7602345]
ranking_edgeacc = [3.2855334, 3.2823699, 3.2778193, 3.2590975]
ranking_f1 = [0.2946237, 0.2949401, 0.2952309, 0.2975257]

# ssi
# 0.03 0.05 0.07
ssi_rmse = [8.3077402, 8.31106, 8.3268026]
ssi_edgecomp = [17.2556566, 17.069636, 16.9474563]
ssi_edgeacc = [3.2664606, 3.2506271, 3.2381389]
ssi_f1 = [0.2977798, 0.300246, 0.3023258]

# plt.figure()
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, layout='constrained', figsize=(12,3))

ax1.axvline(x=baseline_rmse, color='b', linestyle=':')
ax1.axhline(y=baseline_edgecomp, color='b', linestyle=':')
ax1.plot(ranking_rmse, ranking_edgecomp, label='rank', marker='+', color='orange')
ax1.plot(ssi_rmse, ssi_edgecomp, label='ssi', marker='+', color='blue')

ax1.set_xlabel('RMSE')
ax1.set_ylabel('EdgeComp')

ax2.axvline(x=baseline_rmse, color='b', linestyle=':')
ax2.axhline(y=baseline_edgeacc, color='b', linestyle=':')
ax2.plot(ranking_rmse, ranking_edgeacc, label='rank', marker='+', color='orange')
ax2.plot(ssi_rmse, ssi_edgeacc, label='ssi', marker='+', color='blue')
ax2.set_xlabel('RMSE')
ax2.set_ylabel('EdgeAcc')

ax3.axvline(x=baseline_rmse, color='b', linestyle=':')
ax3.axhline(y=baseline_f1, color='b', linestyle=':')
ax3.plot(ranking_rmse, ranking_f1, label='rank', marker='+', color='orange')
ax3.plot(ssi_rmse, ssi_f1, label='ssi', marker='+', color='blue')
ax3.set_xlabel('RMSE')
ax3.set_ylabel('F1')

handles, labels = plt.gca().get_legend_handles_labels()
fig.legend(handles, labels, loc='outside right center')
# plt.tight_layout()
plt.savefig('./work_dir/results.png')
    