import numpy as np

def task2vec(feature_extractor, dataset):
    '''
    Parameters:
        feature_extractor (nn.Module): backbone for FIM probe
        dataset (Dataset): dataset to fit FIM classifier and compute Task2Vec embedding
    Returns:
        embedding (Tensor): FIM Diagonal
    '''
    ## Add classifier head (randomly initialized) to FE
    FIM_probe = nn.Sequential(feature_extractor, nn.Linear())
    feature_extractor.freeze()

    fit_FIM_classifier(FIM_probe, dataset)
    return compute_FIM_diagonal(FIM_probe, dataset)

def run_CL_sequence(N, datasets, model):
    '''
    Parameters:
        N (int): Number of tasks
        datasets (list): train and test dataset for each task
        model (nn.Module): feature extractor + linear classifier

    Returns:
        cl_accs (list): CL accuracy after each new task
        mean_fgts (list): mean forgetting after each new task
        sims (list): cosine sim of task2vec embedding of old tasks and next task to be learned
    '''

    accs = np.zeros(N, N)  # accs[i][j] is the test accuracy of task j after learning task i
    cumulative_sim = 0

    for task_idx in range(N):
        ## Compute Task2Vec embeddings
        if task_idx > 0:
            old_task_vec = task2vec(model.feature_extractor, datasets[:task_idx].train)
            new_task_vec = task2vec(model.feature_extractor, datasets[task_idx].train)
            cumulative_sim += cosine_sim(old_task_vec, new_task_vec)
            sims.append(cur_sim)
        
        ## Learn new task
        model.classifier.extend_head()
        train(model, datasets[task_idx].train)

        ## Evaluate test accuracy on tasks that have been learned
        for prev_task_idx in range(task_idx + 1):
            accs[task_idx][prev_task_idx] = eval(model, datasets[prev_task_idx].test)

        ## Compute CL Metrics
        if task_idx > 0:
            cur_cl_acc = np.mean(accs[task_idx][:task_idx + 1])
            cl_accs.append(cur_cl_acc)

            cur_mean_fgt = np.mean([accs[i][i] - accs[task_id][i] for i in range(task_idx)])
            mean_fgts.append(cur_mean_fgt)

    return cl_accs, mean_fgts, sims
