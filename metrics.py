import torch
import numpy as np
from models import TasksimModel
from scipy.special import softmax
from scipy.stats import entropy
from scipy.spatial.distance import cosine

def run_avg(old, old_count, new, new_count):
    total = old_count + new_count
    return (old_count/total)*old + (new_count/total)*new

# ## From https://stackoverflow.com/questions/44549369/kullback-leibler-divergence-from-gaussian-pm-pv-to-gaussian-qm-qv
# def kl_mvn(m0, S0, m1, S1):
#     """
#     Kullback-Liebler divergence from Gaussian pm,pv to Gaussian qm,qv.
#     Also computes KL divergence from a single Gaussian pm,pv to a set
#     of Gaussians qm,qv.
    

#     From wikipedia
#     KL( (m0, S0) || (m1, S1))
#          = .5 * ( tr(S1^{-1} S0) + log |S1|/|S0| + 
#                   (m1 - m0)^T S1^{-1} (m1 - m0) - N )
#     """

#     S0 = np.diag(np.diag(S0))
#     S1 = np.diag(np.diag(S1))
#     # store inv diag covariance of S1 and diff between means
#     N = m0.shape[0]
#     iS1 = np.linalg.inv(S1)
#     diff = m1 - m0

#     # kl is made of three terms
#     tr_term = np.trace(iS1 @ S0)
#     # np.sum(np.log(S1)) - np.sum(np.log(S0))
#     det_term = np.log(np.linalg.det(S1)/np.linalg.det(S0))
#     # np.sum( (diff*diff) * iS1, axis=1)
#     quad_term = diff.T @ np.linalg.inv(S1) @ diff
#     #print(tr_term,det_term,quad_term)
#     return .5 * (tr_term + det_term + quad_term - N)


def kl_mvn(m0, S0, m1, S1):
    def kl_svn(m0, S0, m1, S1):
        return np.log(np.sqrt(S1)/np.sqrt(S0)) + 0.5*((S0 + (m0 - m1)**2)/S1) - 0.5
    S0 = np.diag(S0)
    S1 = np.diag(S1)
    N = m0.shape[0]
    return np.mean([kl_svn(m0[i], S0[i], m1[i], S1[i]) for i in range(N)])


def wass_mvn(m0, S0, m1, S1):
    def wass_svn(m0, S0, m1, S1):
        return np.sqrt((m0 - m1)**2 + S0 + S1 - 2*np.sqrt(S0*S1))
    S0 = np.diag(S0)
    S1 = np.diag(S1)
    N = m0.shape[0]
    return np.mean([wass_svn(m0[i], S0[i], m1[i], S1[i]) for i in range(N)])

def compute_per_task_metrics(model: TasksimModel, data_loader):
    class_stats = dict()
    max_logit = 0
    max_prob = 0
    ent = 0
    count = 0

    with torch.no_grad():
        for inputs, *labels in data_loader:
            labels = labels[0]
            inputs, labels = inputs.to(model.device), labels.to(model.device)
            features = model.encode_features(inputs) ## B_size x dims
            outs = model.classifier(features)        ## B_size x num_classes
            features = features.cpu().numpy()
            outs = outs.cpu().numpy()
            labels = labels.cpu().numpy()

            ## KL Metric
            for k in np.unique(labels):
                if k not in class_stats:
                    class_stats[k] = {'mean': 0, 'cov': 0, 'counts': 0}
                cur_counts = (labels == k).sum()
                total_counts = class_stats[k]['counts'] + cur_counts
                class_features = features[labels == k]
                cur_mean = np.mean(class_features, axis=0)
                cur_cov = np.cov(class_features, rowvar=False)
                class_stats[k]['mean'] = run_avg(class_stats[k]['mean'], class_stats[k]['counts'], cur_mean, cur_counts)
                class_stats[k]['cov'] = run_avg(class_stats[k]['cov'], class_stats[k]['counts'], cur_cov, cur_counts)
                class_stats[k]['counts'] = total_counts
            
            ## Max log metric
            cur_max_logit = np.max(outs, axis=1).mean()
            max_logit = run_avg(max_logit, count, cur_max_logit, outs.shape[0])

            ## Max prob metric
            probs = softmax(outs, axis=1)
            cur_max_prob = np.max(probs, axis=1).mean()
            max_prob = run_avg(max_prob, count, cur_max_prob, outs.shape[0])

            ## Entropy metric
            cur_ent = entropy(probs, axis=1).mean()
            ent = run_avg(ent, count, cur_ent, outs.shape[0])

            count += outs.shape[0]
        
    return {
        'class_stats': class_stats,
        'max_prob': max_prob,
        'max_logit': max_logit,
        'entropy': ent
    }


def compute_metrics(model: TasksimModel, old_data_loader, new_data_loader):
    old_metrics = compute_per_task_metrics(model, old_data_loader)
    new_metrics = compute_per_task_metrics(model, new_data_loader)

    metrics = {}

    metrics['max_prob_diff'] = old_metrics['max_prob'] - \
        new_metrics['max_prob']
    metrics['max_prob_ratio'] = old_metrics['max_prob']/new_metrics['max_prob']
    metrics['max_logit_diff'] = old_metrics['max_logit'] - \
        new_metrics['max_logit']
    metrics['max_logit_ratio'] = old_metrics['max_logit'] / \
        new_metrics['max_logit']
    metrics['entropy_diff'] = old_metrics['entropy'] - new_metrics['entropy']
    metrics['entropy_ratio'] = old_metrics['entropy']/new_metrics['entropy']

    kl_div = 0
    wass_dist = 0
    for i in old_metrics['class_stats'].keys():
        for j in new_metrics['class_stats'].keys():
            kl_div += kl_mvn(
                old_metrics['class_stats'][i]['mean'],
                old_metrics['class_stats'][i]['cov'],
                new_metrics['class_stats'][j]['mean'],
                new_metrics['class_stats'][j]['cov'])
            wass_dist += wass_mvn(
                old_metrics['class_stats'][i]['mean'],
                old_metrics['class_stats'][i]['cov'],
                new_metrics['class_stats'][j]['mean'],
                new_metrics['class_stats'][j]['cov'])
    
    kl_div=kl_div / (len(old_metrics['class_stats']) * len(new_metrics['class_stats']))
    metrics['kl_div']=kl_div

    wass_dist = wass_dist / \
        (len(old_metrics['class_stats']) * len(new_metrics['class_stats']))
    metrics['wass_dist'] = wass_dist
    return metrics
