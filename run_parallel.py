import subprocess
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=12)
parser.add_argument("--device", type=str, default='cuda:0')
parser.add_argument("--task", type=str, default='drop')
parser.add_argument("--MAX_B", type=int, default=8)
parser.add_argument("--MAX_ORDER", type=int, default=4)
parser.add_argument("--NUM_EXPLAIN", type=int, default=25)
parser.add_argument("--num_test_samples", type=int, default=10000)
parser.add_argument("--use_cache", type=bool, default=True)
parser.add_argument("--run_sampling", type=bool, default=False)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--num_processes", type=int, default = 1)
args = parser.parse_args()


[subprocess.Popen(['python3', 'experiments/measure_r2.py', '--seed', str(i + args.seed), '--device', f'cuda:{i}', '--task', args.task, '--MAX_B', 
                    str(args.MAX_B), '--MAX_ORDER', str(args.MAX_ORDER), '--NUM_EXPLAIN', str(args.NUM_EXPLAIN), '--num_test_samples', str(args.num_test_samples),
                    '--use_cache', str(args.use_cache), '--batch_size', str(args.batch_size), '--run_sampling', str(args.run_sampling)]) for i in range(args.num_processes)]
# for i in range(torch.cuda.device_count()):
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