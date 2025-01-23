import numba
import numpy as np
import cProfile, argparse
from spectral_explain.models.modelloader import HotPotQAModel, QAModel
from spectral_explain.dataloader import get_dataset
from experiment_utils import run_sampling, get_and_evaluate_reconstruction

SAVE_DIR = 'experiments/results'

def main(seed=12, device='cuda:0', task='drop', MAX_B=8, MIN_B=8, MAX_ORDER=4, 
    num_explain=1, num_test_samples=10000, t = 5, batch_size=512, verbose=True):
    print("Loading model and explicands")
    explicands = get_dataset(task, num_explain = num_explain, seed = seed)
    print(f"Finished loading explicands")
    if task == 'hotpotqa':
        model = HotPotQAModel(device = device)
    else:
        model = QAModel(device = device)
    model.batch_size = batch_size
    print("Finished loading model and explicands")
    np.random.seed(seed)
    for explicand in explicands:
        sample_id = explicand['id']
        n = model.set_explicand(explicand)
        sampling_function = lambda X: model.inference(X)
        save_dir = f'{SAVE_DIR}/{task}/{sample_id}'
        #run_sampling(model=model, explicand=explicand, sampling_function=sampling_function, Bs = [3,4], n = n, save_dir = save_dir, 
        #             order = MAX_ORDER, num_test_samples = num_test_samples, verbose = verbose)
        get_and_evaluate_reconstruction(explicand = explicand, Bs = [3,4], max_order = MAX_ORDER, save_dir = save_dir, t = t)
        

if __name__ == "__main__":
    print("Starting main function")
    profiler = cProfile.Profile()
    profiler.enable()
    numba.set_num_threads(2)
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument("--task", type=str, default='hotpotqa')
    parser.add_argument("--MAX_B", type=int, default=3)
    parser.add_argument("--MIN_B", type=int, default=3)
    parser.add_argument("--MAX_ORDER", type=int, default=4)
    parser.add_argument("--num_test_samples", type=int, default=100)
    parser.add_argument("--run_sampling", type=bool, default=False)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--verbose", type=bool, default=True)
    parser.add_argument("--t", type=int, default=5)
    args = parser.parse_args()
    main(seed=args.seed, device=args.device, task=args.task, num_explain=1, 
        MAX_B=args.MAX_B, MIN_B=args.MIN_B, MAX_ORDER=args.MAX_ORDER, 
        num_test_samples=args.num_test_samples, batch_size=args.batch_size, 
        t = args.t, verbose=args.verbose)