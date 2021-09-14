import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics


def show_dist(train_dist, train_is_diff, train_is_ood, val_dist, val_is_diff, val_is_ood, plot = True):
    #train
    train_dist = np.squeeze(train_dist.cpu().detach().numpy())
    train_is_diff = np.squeeze(train_is_diff.cpu().numpy())
    train_is_ood = np.squeeze(train_is_ood.cpu().numpy())
    tr_same_dist = [train_dist[i] for i in range(len(train_dist)) if train_is_diff[i] == 0]
    tr_diff_seen_dist = [train_dist[i] for i in range(len(train_dist)) if (train_is_diff[i] == 1 and train_is_ood[i] ==0)]
    tr_diff_ood_dist = [train_dist[i] for i in range(len(train_dist)) if (train_is_diff[i] == 1 and train_is_ood[i] ==1)]
    tr_mean_same_dist, tr_mean_diff_seen_dist, tr_mean_diff_ood_dist = np.mean(tr_same_dist),np.mean(tr_diff_seen_dist),(np.mean(tr_diff_ood_dist) if tr_diff_ood_dist!=[] else 'NA')
    print("Distance distribution of last training batch:\nmean_same = {}, mean_diff_seen = {}, mean_diff_ood = {}"
                  .format(tr_mean_same_dist, tr_mean_diff_seen_dist, tr_mean_diff_ood_dist))
    
    #val
    val_dist = np.squeeze(val_dist.cpu().detach().numpy())
    val_is_diff = np.squeeze(val_is_diff.cpu().numpy())
    val_is_ood = np.squeeze(val_is_ood.cpu().numpy())
    val_same_dist = [val_dist[i] for i in range(len(val_dist)) if val_is_diff[i] == 0]
    val_diff_seen_dist = [val_dist[i] for i in range(len(val_dist)) if (val_is_diff[i] == 1 and val_is_ood[i] ==0)]
    val_diff_ood_dist = [val_dist[i] for i in range(len(val_dist)) if (val_is_diff[i] == 1 and val_is_ood[i] ==1)]
    val_mean_same_dist, val_mean_diff_seen_dist, val_mean_diff_ood_dist = np.mean(val_same_dist),np.mean(val_diff_seen_dist),(np.mean(val_diff_ood_dist) if val_diff_ood_dist!=[] else 'NA')

    print("Distance distribution of last validation batch:\nmean_same = {}, mean_diff_seen = {}, mean_diff_ood = {}"
                  .format(val_mean_same_dist, val_mean_diff_seen_dist, val_mean_diff_ood_dist))
    
    if plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
        ax1.hist(tr_same_dist, alpha = 0.5, bins = 20, color='g', label='same')
        ax1.hist(tr_diff_seen_dist, alpha = 0.5, bins = 20, color='b', label='diff_seen')
        ax1.hist(tr_diff_ood_dist, alpha = 0.5, bins = 20, color='r', label='diff_ood')
        ax1.legend()
        ax1.set_title("train - distance")
        ax2.hist(val_same_dist, alpha = 0.5, bins = 20, color='g', label='same')
        ax2.hist(val_diff_seen_dist, alpha = 0.5, bins = 20, color='b', label='diff_seen')
        ax2.hist(val_diff_ood_dist, alpha = 0.5, bins = 20, color='r', label='diff_ood')
        ax2.legend()
        ax2.set_title("validation - distance")
        plt.show()    

    return tr_mean_same_dist, tr_mean_diff_seen_dist, tr_mean_diff_ood_dist, val_mean_same_dist, val_mean_diff_seen_dist, val_mean_diff_ood_dist


def show_dist_tr(train_dist, train_is_diff, train_is_ood, plot = True):
    #train
    train_dist = np.squeeze(train_dist.cpu().detach().numpy())
    train_is_diff = np.squeeze(train_is_diff.cpu().numpy())
    train_is_ood = np.squeeze(train_is_ood.cpu().numpy())
    tr_same_dist = [train_dist[i] for i in range(len(train_dist)) if train_is_diff[i] == 0]
    tr_diff_seen_dist = [train_dist[i] for i in range(len(train_dist)) if (train_is_diff[i] == 1 and train_is_ood[i] ==0)]
    tr_diff_ood_dist = [train_dist[i] for i in range(len(train_dist)) if (train_is_diff[i] == 1 and train_is_ood[i] ==1)]
    tr_mean_same_dist, tr_mean_diff_seen_dist, tr_mean_diff_ood_dist = np.mean(tr_same_dist),np.mean(tr_diff_seen_dist),(np.mean(tr_diff_ood_dist) if tr_diff_ood_dist!=[] else 'NA')
    print("Distance distribution of last training batch:\nmean_same = {}, mean_diff_seen = {}, mean_diff_ood = {}"
                  .format(tr_mean_same_dist, tr_mean_diff_seen_dist, tr_mean_diff_ood_dist))
    
    if plot:
        print("train - distance")
        plt.hist(tr_same_dist, alpha = 0.5, bins = 20, color='g', label='same')
        plt.hist(tr_diff_seen_dist, alpha = 0.5, bins = 20, color='b', label='diff_seen')
        plt.hist(tr_diff_ood_dist, alpha = 0.5, bins = 20, color='r', label='diff_ood')
        plt.legend()
        plt.show()    

    return tr_mean_same_dist, tr_mean_diff_seen_dist, tr_mean_diff_ood_dist


def show_auroc(label, score, display = True):
    # calculate auroc
    # This function requires the true binary value and the target scores 
    # which can either be probability estimates of the positive class, confidence values, or binary decisions
    #fpr, tpr, thresholds = metrics.roc_curve(is_ood_list, ood_score_list, pos_label=1)
    fpr, tpr, thresholds = metrics.roc_curve(label, score, pos_label=1)
    auroc = metrics.auc(fpr, tpr)

    # plot roc curve
    if display:
        print("auroc:", auroc)
        plt.title('Receiver Operating Characteristic')
        plt.plot(fpr, tpr, 'b', label = 'AUC = %0.4f' % auroc)
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()
    
    return auroc
