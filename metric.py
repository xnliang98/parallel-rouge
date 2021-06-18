import rouge

def compute_metrics(hypothesis, references, aggregator="Avg"):
    apply_avg = aggregator == 'Avg'
    apply_best = aggregator == 'Best'
    evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l', 'rouge-w'],
                           max_n=3,
#                            limit_length=True,
#                            length_limit=100, # default 665
#                            length_limit_type='words',
                           apply_avg=apply_avg,
                           apply_best=apply_best,
                           alpha=0.5, # Default F1_score
                           weight_factor=1.0, # used fro rouge-w, if you want to compute rouge-w, you can use 1.2, which lead to the rouge-l will be wrong [this is a bug] 
                           stemming=True)
    scores = evaluator.get_scores(hypothesis, references)
    
    return scores

def prepare_results(metric, p, r, f):
    return '\t{}:\t{}: {:5.2f}\t{}: {:5.2f}\t{}: {:5.2f}'.format(metric.upper(), 'P', 100.0 * p, 'R', 100.0 * r, 'F1', 100.0 * f)

def parallel_process(data):
    all_hypothesis, all_references = data
    return compute_metrics(all_hypothesis, all_references)

def compute_rouge_parallel(all_hypothesis, all_references, aggregator="Avg", ncpus=20):
    total_num = len(all_hypothesis)
    block_size = total_num // ncpus
    
    data = [(all_hypothesis[i * block_size: (i+1) * block_size], all_references[i * block_size: (i+1) * block_size]) for i in range(ncpus)]
    
    from multiprocessing import Pool
    with Pool(ncpus) as pool:
        scores_lst = pool.map(parallel_process, data)

    micro_scores = scores_lst[0]
    scores_lst.pop(0)
    assert len(scores_lst) == ncpus - 1
    for scores in scores_lst:
        for key, value in sorted(scores.items(), key=lambda x: x[0]):
            micro_scores[key]['f'] += value['f']
            micro_scores[key]['p'] += value['p']
            micro_scores[key]['r'] += value['r']
    for key, value in sorted(micro_scores.items(), key=lambda x: x[0]):
        print(prepare_results(key, value['p'] / ncpus, value['r'] / ncpus, value['f'] / ncpus))
    return micro_scores
        
  
compute_rouge_parallel(all_hypothesis, all_references)