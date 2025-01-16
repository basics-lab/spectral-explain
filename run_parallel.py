import subprocess; [subprocess.Popen(['python3', 'experiments/measure_r2.py', '--seed', str(i), '--device', f'cuda:{i}']) for i in range(4)]
