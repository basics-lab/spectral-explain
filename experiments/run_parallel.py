import subprocess
import torch
import concurrent.futures
import argparse
import os
from collections import deque


def run_job(job_id, batch_size):
    device_id = job_id % torch.cuda.device_count()
    device = f'cuda:{device_id}'
    command = f'python3 experiments/collect_data.py --seed {job_id} --device {device} --task {args.task} --MAX_B {args.MAX_B} --MAX_ORDER {args.MAX_ORDER} --num_test_samples {args.num_test_samples} --batch_size {batch_size} --verbose {args.verbose}'
    log_dir = f"experiments/logs/{args.task}"
    os.makedirs(log_dir, exist_ok=True)
    output_file = f"experiments/logs/{args.task}/job_{job_id}_output.txt"  # Output file for the job
    error_file = f"experiments/logs/{args.task}/job_{job_id}_error.txt"   # Error file for the job
    print(f"Starting Job {job_id} on GPU {device_id}: {command}")
    with open(output_file, 'w') as outfile, open(error_file, 'w') as errfile:
        result = subprocess.run(command, shell=True, text=True, stdout=outfile, stderr=errfile)
    print(f"Finished Job {job_id} with return code {result.returncode}")
    return result.returncode

def process_completed_jobs(done, futures, job_ids):
    for future, job_id, batch_size in [(f, j, size) for f, j, size in futures if f in done]:
        try:
            result = future.result()  # Get the result (return code)
            if result != 0:  # If the job failed, re-add to the front
                print(f"Job {job_id} failed. Re-adding to the front of the queue.")
                job_ids.append((job_id, batch_size//2))
            else:
                print(f"Job {job_id} completed successfully with return code: {result}")
        except Exception as e:
            print(f"Job {job_id} raised an exception: {e}. Re-adding to the front of the queue.")
            job_ids.append((job_id, batch_size))
    # Update the futures list with only the jobs still running
    return [(f, j, batch_size) for f, j, batch_size in futures if f not in done]    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_explain", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--task", type=str, default='drop')
    parser.add_argument("--MAX_B", type=int, default=8)
    parser.add_argument("--MIN_B", type=int, default=3)
    parser.add_argument("--MAX_ORDER", type=int, default=4)
    parser.add_argument("--num_test_samples", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--verbose", type=bool, default=True)
    args = parser.parse_args()
    num_jobs = args.num_explain
    successfully_completed = 0
    job_ids = deque([(i + args.seed, args.batch_size) for i in range(args.num_explain)])




    with concurrent.futures.ProcessPoolExecutor(max_workers=torch.cuda.device_count()) as executor:
        futures = []
        
        # If the number of jobs is greater than the number of GPUs, wait for a GPU to finish
        while job_ids or futures:
            while len(futures) < torch.cuda.device_count():
                job_id, batch_size = job_ids.popleft()
                future = executor.submit(run_job, job_id, batch_size)
                futures.append((future, job_id, batch_size))
            
            done, not_done = concurrent.futures.wait([f[0] for f in futures], return_when=concurrent.futures.FIRST_COMPLETED)
            futures = process_completed_jobs(done, futures, job_ids)
            



                    
        #             done, not_done = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
        #             for future in done:
        #                     try:
        #                         returncode = future.result()
        #                     if returncode == 0:
        #                         successfully_completed += 1
        #             except Exception as e:
        #                 print(f"Job {job_id} raised an exception: {e}")
        #         futures = list(not_done)

        #     future = executor.submit(run_job, job_id)
        #     futures.append(future)
        
        # concurrent.futures.wait(futures)

        #     futures.append(executor.submit(run_job, job_id))
        # for future in concurrent.futures.as_completed(futures):
        #     job_id = futures[future]
        #     try:
        #         result = future.result()  # Get the result of the job
        #         print(f"Job {job_id} completed successfully with result: {result}")
        #     except Exception as e:
        #         print(f"Job {job_id} raised an exception: {e}")


# for i in range(args.num_explain):
#     device = f'cuda:{i}'
#     log_filename = f"log_gpu_{i}.txt"  # Define log file for each GPU
#     with open(log_filename, 'w') as log_file:
#         subprocess.Popen(
#             ['python3', 'collect_data.py', '--seed', str(i + args.seed), '--device', f'cuda:{i}', '--task', args.task, '--MAX_B', 
#             str(args.MAX_B), '--MAX_ORDER', str(args.MAX_ORDER), '--num_test_samples', str(args.num_test_samples),
#             '--batch_size', str(args.batch_size), '--verbose', str(args.verbose)],
#             stdout=log_file,  # Redirect stdout to the log file
#             stderr=log_file  # Redirect stderr to the same log file
#         )

#     log_filename = f"log_gpu_{i}.txt"  # Define log file for each GPU
#     with open(log_filename, 'w') as log_file:
#         subprocess.Popen(
#             ['python3', 'experiments/measure_r2.py',
#              '--seed', str(i + args.seed),
#              '--device', f'cuda:{i}',
#              '--task', args.task,
#              '--MAX_B', str(args.MAX_B),
#              '--MAX_ORDER', str(args.MAX_ORDER),
#              '--NUM_EXPLAIN', str(args.NUM_EXPLAIN),
#              '--num_test_samples', str(args.num_test_samples),
#              '--use_cache', str(args.use_cache),
#              '--batch_size', str(args.batch_size),
#              '--collect_data', str(args.collect_data)],
#             stdout=log_file,  # Redirect stdout to the log file
#             stderr=log_file  # Redirect stderr to the same log file
#         )