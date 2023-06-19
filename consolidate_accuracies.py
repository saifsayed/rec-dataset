import numpy as np

jump=75
threshold=10
dataset = 'gtea'
dataset2split = {'50salads': 6, 'gtea': 5, 'mpii_cooking2': 2}
acc_split = np.load('./results/{}/{}_{}_split_{}/preds_{}_{}_split_{}.npy'.format(dataset, jump, threshold, 1, jump, threshold, 1),allow_pickle=True)
num_runs = np.shape(acc_split)[0]

all_splits = []
for split_id in range(1, dataset2split[dataset]):
    acc_split = np.load('./results/{}/{}_{}_split_{}/preds_{}_{}_split_{}.npy'.format(dataset, jump, threshold, split_id, jump, threshold, split_id),allow_pickle=True)
    all_splits.append(acc_split)
all_splits = np.asarray(all_splits)
finalval = (np.mean(all_splits,axis=0))[0]

print('Final Test Result for all splits\n')
print('F1@0.10: {:.2f}\nF1@0.25: {:.2f}\nF1@0.50: {:.2f}\nEdit: {:.2f}\nAcc: {:.2f}'.format(finalval[0], finalval[1],finalval[2],finalval[3],finalval[4]))
