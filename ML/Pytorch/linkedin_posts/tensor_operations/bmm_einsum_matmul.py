import torch
import torch.utils.benchmark as benchmark

# Sample input batched matrices dimensions.
n_batch, n, m, k = 10, 64, 128, 32

# Setup code as a string to be reused across benchmarks
setup_code = (
    f'import torch; '
    f'x = torch.randn({n_batch}, {n}, {m}); '
    f'y = torch.randn({n_batch}, {m}, {k})')

# Number of threads from torch, reused in all timers.
num_threads = torch.get_num_threads()

# A list of methods and their stmt strings for the benchmark
methods = [
    ('bmm', 'torch.bmm(x, y)'),
    ('matmul', 'torch.matmul(x, y)'),
    ('einsum', "torch.einsum('bnm,bmk->bnk', x, y)"),
]

# Run each benchmark for a number of times to ensure measurement stability
num_runs = 100

# Create benchmark objects and run them, collecting the results.
results = [
    benchmark.Timer(
        stmt=stmt,
        setup=setup_code,
        num_threads=num_threads,
        label="Batched Matrix Multiplication",
        sub_label=f"Method: {label}",
        description=f"{n_batch}x{n}x{m}x{k}",
    ).timeit(num_runs)
    for label, stmt in methods
]

# Group the results into a Compare object and print the results table.
benchmark.Compare(results).print()
