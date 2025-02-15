import numba
import numpy as np
import pandas as pd
from tqdm import tqdm
import dill as pickle
import os, shutil, cProfile, pstats, gc, argparse
from spectral_explain.models.modelloader import get_model
from spectral_explain.support_recovery import sampling_strategy, get_num_samples
from spectral_explain.utils import estimate_r2
from experiment_utils import get_and_evaluate_reconstruction
from math import comb
from tqdm import tqdm
from joblib import Parallel, delayed
SAVE_DIR = f'experiments/results/'


def measure_r2(task, MAX_ORDER):
    pass

def get_num_samples(task,save_dir):
    results_dir = f'{save_dir}/{task}'
    num_samples = 0
    for subdir in os.listdir(results_dir):
        if subdir == 'r2_results.pkl':
            continue
        subdir_path = os.path.join(results_dir, subdir)
        if 'lime_b8_order1.pickle' in os.listdir(subdir_path):
            num_samples += 1
    return num_samples


def process_explicand(results_dir, subdir, explicand, Bs, max_order, t):
    subdir_path = os.path.join(results_dir, subdir)
    reconstruction_dict, r2_results = get_and_evaluate_reconstruction(explicand = explicand, Bs = Bs, save_dir = subdir_path, max_order = max_order, t = t)
    
    print(f'Finished reconstruction for explicand {explicand["id"]}')
    return reconstruction_dict, r2_results

def main(task = 'drop', max_order = 4, Bs = [4,6,8], t = 5, save_dir = 'experiments/results'):
    count = 0
    all_results = []
    reg_methods = [('linear', i) for i in range(1,max_order+1)] + [('lasso', i) for i in range(1,max_order+1)] + [('faith_banzhaf', i) for i in range(1,max_order+1)]
    qsft_methods = [('qsft_hard', 0), ('qsft_soft', 0)]
    shap_methods = [('SV', 1)] +  [('FSII', i) for i in range(1,max_order+1)] + [('STII', i) for i in range(1,max_order+1)]
    lime_methods = [('lime', 1)]
    ordered_methods = reg_methods + qsft_methods + shap_methods  + lime_methods
    num_samples = get_num_samples(task, save_dir)
    results_dir = f'{save_dir}/{task}'
    results = {
        "samples": np.zeros((num_samples, len(Bs))),
        "methods": {f'{method}_{order}': {'time': np.zeros((num_samples, len(Bs))),
                                          'test_r2': np.zeros((num_samples, len(Bs)))}
                    for method, order in ordered_methods}
    }
    i = 0
    # Get list of subdirs and explicands to process
    explicand_list = []
    for subdir in os.listdir(results_dir):
        if subdir == 'r2_results.pkl':
            continue
        subdir_path = os.path.join(results_dir, subdir)
        if 'lime_b8_order1.pickle' in os.listdir(subdir_path):
            explicand = pickle.load(open(os.path.join(subdir_path, 'explicand_information.pickle'), 'rb'))
            explicand_list.append((subdir, explicand))
    

    # Process explicands in parallel
    
    reconstruction_results = []
    r2_results = []
    results_list = Parallel(n_jobs=45)(delayed(process_explicand)(results_dir, subdir, explicand, Bs, max_order, t) for subdir, explicand in explicand_list)

    for result in results_list:
        reconstruction_results.append((result[0][('qsft_soft_0', 8)], result[0]['explicand']['n']))
        r2_results.append(result[1])
    with open(f'{save_dir}/{task}/r2_results.pkl', 'wb') as handle:
        pickle.dump(r2_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open(f'experiments/processed_results/reconstructions/reconstruction_results_{task}.pkl', 'wb') as handle:
        pickle.dump(reconstruction_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    print("Starting main function")
    profiler = cProfile.Profile()
    profiler.enable()
    numba.set_num_threads(8)
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default='drop')
    parser.add_argument("--Bs", type=int, nargs='+', default=[4, 6, 8])
    parser.add_argument("--t", type=int, default=5)
    parser.add_argument("--MAX_ORDER", type=int, default=4)
    
    args = parser.parse_args()
    main(task = args.task, max_order = args.MAX_ORDER, Bs = [4,6,8], t = args.t, save_dir = SAVE_DIR)
     # try:
    #     with open(os.path.join(subdir_path, 'reconstruction_dict.pickle'), 'rb') as handle:
    #         reconstruction_dict = pickle.load(handle)
    #     with open(os.path.join(subdir_path, 'r2_results.pkl'), 'rb') as handle:
    #         r2_results = pickle.load(handle)
    # except Exception as e:
    #     print(f'Explicand {explicand["id"]} not cached. Running reconstruction.')