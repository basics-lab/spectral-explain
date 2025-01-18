import numba
import numpy as np
import pandas as pd
from tqdm import tqdm
import os, shutil, cProfile, pstats, gc, argparse
from spectral_explain.models.modelloader import get_model
from experiment_utils import run_sampling


def main(seed=12, device='cuda:0', task='drop', MAX_B=8, MIN_B=8, MAX_ORDER=4, 
    num_test_samples=10000, batch_size=512, verbose=True):
    print("Loading model and explicands")
    explicands, model = get_model(task = task, num_explain = 1, device = device, seed = seed)
    model.batch_size = batch_size
    print("Finished loading model and explicands")
    np.random.seed(seed)
    for explicand in explicands:
        sample_id = explicand['id']
        n = model.set_explicand(explicand)
        sampling_function = lambda X: model.inference(X)
        save_dir = f'/scratch/users/{os.getenv("USER")}/results/{task}/{sample_id}'
        run_sampling(explicand, sampling_function, b = MAX_B, n = n, save_dir = save_dir, 
                     order = MAX_ORDER, num_test_samples = num_test_samples, verbose = verbose)
        

if __name__ == "__main__":
    print("Starting main function")
    profiler = cProfile.Profile()
    profiler.enable()
    numba.set_num_threads(8)
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument("--task", type=str, default='drop')
    parser.add_argument("--MAX_B", type=int, default=8)
    parser.add_argument("--MIN_B", type=int, default=8)
    parser.add_argument("--MAX_ORDER", type=int, default=4)
    parser.add_argument("--num_test_samples", type=int, default=100)
    parser.add_argument("--use_cache", type=bool, default=True)
    parser.add_argument("--run_sampling", type=bool, default=False)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--verbose", type=bool, default=True)
    main()