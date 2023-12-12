import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

def measure_performance(matrix_type, n, device, num_runs=100):
    matrix = torch.randn(n, n, dtype=matrix_type, device=device)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # Warming up
    for _ in range(5):
        torch.matmul(matrix, matrix)

    # Measure performance
    start_event.record()
    for _ in range(num_runs):
        torch.matmul(matrix, matrix)
    end_event.record()

    # Synchronizes events to ensure the time is measured correctly
    torch.cuda.synchronize()
    return start_event.elapsed_time(end_event) / num_runs

n = 2**11
num_runs = 100

# Dictionary to store execution time for different data types
execution_times = {
    'float16': measure_performance(torch.float16, n, device, num_runs),
    'float32': measure_performance(torch.float32, n, device, num_runs),
    'float64': measure_performance(torch.float64, n, device, num_runs),
}

print(f'Execution time ({device}):')
for dtype, time in execution_times.items():
    print(f'{dtype}: {time:.6f} ms')
