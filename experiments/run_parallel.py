import subprocess
import torch
import concurrent.futures
import argparse
import os
from collections import deque


def run_job(job_id, batch_size):
    device_id = job_id % torch.cuda.device_count()
    device = f'cuda:{device_id}'
    Bs_list = ' '.join(map(str, args.Bs))
    command = f'python3 experiments/collect_data.py --seed {job_id} --device {device} --task {args.task} --MAX_ORDER {args.MAX_ORDER} --num_test_samples {args.num_test_samples} --batch_size {batch_size} --t {args.t}'
    log_dir = f"experiments/logs/{args.task}"
    os.makedirs(log_dir, exist_ok=True)
    output_file = f"experiments/logs/{args.task}/job_{job_id}_output.txt"  # Output file for the job
    error_file = f"experiments/logs/{args.task}/job_{job_id}_error.txt"   # Error file for the job
    print(f"Starting Job {job_id} on GPU {device_id}: {command}")
    with open(output_file, 'w') as outfile, open(error_file, 'w') as errfile:
        #result = subprocess.run(command, shell=True, text=True, stdout=outfile, stderr=errfile)
        result = subprocess.run(command, shell=True, text=True, stdout=None, stderr=None)

    print(f"Finished Job {job_id} with return code {result.returncode}")
    return result.returncode

def process_completed_jobs(done, futures, job_ids):
    for future, job_id, batch_size in [(f, j, size) for f, j, size in futures if f in done]:
        try:
            result = future.result()  # Get the result (return code)
            if result != 0:  # If the job failed, re-add to the front
                print(f"Job {job_id} failed. Re-adding to the back of the queue.")
                job_ids.appendleft((job_id, batch_size//2))
            else:
                print(f"Job {job_id} completed successfully with return code: {result}")
        except Exception as e:
            print(f"Job {job_id} raised an exception: {e}. Re-adding to the back of the queue.")
    # Update the futures list with only the jobs still running
    return [(f, j, batch_size) for f, j, batch_size in futures if f not in done]    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_explain", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task", type=str, default='drop')
    parser.add_argument("--MAX_ORDER", type=int, default=4)
    parser.add_argument("--num_test_samples", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--Bs", type=int, nargs='+', default=[4,6,8], help="List of B values for sampling")
    parser.add_argument("--t", type=int, default=5)
    args = parser.parse_args()
    num_jobs = args.num_explain
    successfully_completed = 0
    job_ids = deque([(i + args.seed, args.batch_size) for i in range(args.num_explain)])
    num_workers = torch.cuda.device_count()



    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        
        # If the number of jobs is greater than the number of GPUs, wait for a GPU to finish
        while (len(job_ids) > 0) or futures:
            while len(futures) < num_workers:
                job_id, batch_size = job_ids.popleft()
                future = executor.submit(run_job, job_id, batch_size)
                futures.append((future, job_id, batch_size))
            
            done, not_done = concurrent.futures.wait([f[0] for f in futures], return_when=concurrent.futures.FIRST_COMPLETED)
            futures = process_completed_jobs(done, futures, job_ids)
            
