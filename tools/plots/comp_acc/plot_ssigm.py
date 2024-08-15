
import matplotlib.pyplot as plt



baseline_sqrel = 1.0256434
baseline_rmse = 8.3237575
baseline_edgecomp = 22.1390643
# baseline_edgecomp = 3.4841872
baseline_edgeacc = 3.3948909
baseline_f1 = 0.2699879

# ranking
# 0.0003 0.0005 0.005
ranking_rmse = [8.3064499, 8.3113247, 8.3268827]
ranking_edgecomp = [17.744182, 17.7742125, 17.8874318]
ranking_edgeacc = [3.2835496, 3.2862588, 3.2611528]
ranking_f1 = [0.2948081, 0.2944786 , 0.2974126]
  
# ssi
# 0.01 0.05 0.07
ssi_rmse = [8.2999069, 8.3132484, 8.3228864]
ssi_edgecomp = [17.1986394, 17.05429, 16.899661]
ssi_edgeacc = [3.2861413, 3.2505162, 3.2379736]
ssi_f1 = [0.2952367 , 0.3040395 , 0.3023468]
 
 # ssigm
ssi_midas_grad_rmse = [8.3089576, 8.3122255, 8.3174525, 8.3194773, 8.3308413]
ssi_midas_grad_edgecomp = [17.5388561, 16.3886172, 15.416302, 14.803599, 13.9384693]
ssi_midas_grad_edgeacc = [3.1524449, 3.0361059, 3.0008737, 2.9899607, 2.9947226]
ssi_midas_grad_f1 = [0.314006, 0.3377866, 0.3516453, 0.3599078, 0.3674157]



# last stu conv ssi
distill_ssi_midas_grad_rmse = [8.297111, 8.297497, 8.2955459, 8.3061428, 8.3222781]
distill_ssi_midas_grad_edgecomp = [17.4770716, 17.5106175, 17.4845061, 16.6159052, 16.0771619]
distill_ssi_midas_grad_edgeacc = [3.2982864, 3.28905, 3.2814589, 3.2244282, 3.1826144]
distill_ssi_midas_grad_f1 = [0.2924322, 0.2935138, 0.2949085, 0.3054543, 0.312349]


# mid
# distill_ssi_midas_grad_rmse = [8.3108432, 8.3052987, 8.3088623, 8.3059804]
# distill_ssi_midas_grad_edgecomp = [17.8340884, 17.6729527, 17.6578376, 17.6388496]
# distill_ssi_midas_grad_edgeacc = [3.2868075, 3.2845836, 3.286828, 3.2842817]
# distill_ssi_midas_grad_f1 = [0.2945072, 0.2943754, 0.2942278, 0.2939764]

# plt.figure()
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, layout='constrained', figsize=(12,3))

ax1.axvline(x=baseline_rmse, color='gray', linestyle=':')
ax1.axhline(y=baseline_edgecomp, color='gray', linestyle=':')
ax1.plot(ssi_midas_grad_rmse, ssi_midas_grad_edgecomp, label='ssigm', marker='+', color='green', linestyle='--')
ax1.plot(ranking_rmse, ranking_edgecomp, label='rank', marker='+', color='orange', linestyle='--')
ax1.plot(ssi_rmse, ssi_edgecomp, label='ssi', marker='+', color='blue', linestyle='--')
# ax1.plot(off_ssi_midas_grad_rmse, off_ssi_midas_grad_edgecomp, label='offline-ssigm-uncert', marker='+', color='green')
ax1.plot(distill_ssi_midas_grad_rmse, distill_ssi_midas_grad_edgecomp, label='distill', marker='+', color='black')
# ax1.plot(off_ranking_rmse, off_ranking_edgecomp, label='rank', marker='+', color='orange')
# ax1.plot(off_ssi_rmse, off_ssi_edgecomp, label='ssi', marker='+', color='blue')

ax1.set_xlabel('RMSE')
ax1.set_ylabel('EdgeComp')

ax2.axvline(x=baseline_rmse, color='gray', linestyle=':')
ax2.axhline(y=baseline_edgeacc, color='gray', linestyle=':')
ax2.plot(ssi_midas_grad_rmse, ssi_midas_grad_edgeacc, label='online-ssigm', marker='+', color='green', linestyle='--')
ax2.plot(ranking_rmse, ranking_edgeacc, label='online-rank', marker='+', color='orange', linestyle='--')
ax2.plot(ssi_rmse, ssi_edgeacc, label='online-ssi', marker='+', color='blue', linestyle='--')
# ax2.plot(off_ssi_midas_grad_rmse, off_ssi_midas_grad_edgeacc, label='offline-ssigm-uncert', marker='+', color='green')
ax2.plot(distill_ssi_midas_grad_rmse, distill_ssi_midas_grad_edgeacc, label='distill', marker='+', color='black')
# ax2.plot(off_ranking_rmse, off_ranking_edgeacc, label='online-rank', marker='+', color='orange')
# ax2.plot(off_ssi_rmse, off_ssi_edgeacc, label='online-ssi', marker='+', color='blue')
ax2.set_xlabel('RMSE')
ax2.set_ylabel('EdgeAcc')

ax3.axvline(x=baseline_rmse, color='gray', linestyle=':')
ax3.axhline(y=baseline_f1, color='gray', linestyle=':')
ax3.plot(ssi_midas_grad_rmse, ssi_midas_grad_f1, label='online-ssigm', marker='+', color='green', linestyle='--')
ax3.plot(ranking_rmse, ranking_f1, label='online-rank', marker='+', color='orange', linestyle='--')
ax3.plot(ssi_rmse, ssi_f1, label='online-ssi', marker='+', color='blue', linestyle='--')
# ax3.plot(off_ssi_midas_grad_rmse, off_ssi_midas_grad_f1, label='offline-ssigm-uncert', marker='+', color='green')
ax3.plot(distill_ssi_midas_grad_rmse, distill_ssi_midas_grad_f1, label='distill', marker='+', color='black')
# ax3.plot(off_ranking_rmse, off_ranking_f1, label='online-rank', marker='+', color='orange')
# ax3.plot(off_ssi_rmse, off_ssi_f1, label='online-ssi', marker='+', color='blue')
ax3.set_xlabel('RMSE')
ax3.set_ylabel('F1')

handles, labels = plt.gca().get_legend_handles_labels()
fig.legend(handles, labels, loc='outside right center')
# plt.tight_layout()
plt.savefig('./work_dir/results_ssigm.png')
    