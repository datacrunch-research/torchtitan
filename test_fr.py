import torch
import torch.distributed as dist
import os
from datetime import timedelta

local_rank = int(os.environ["LOCAL_RANK"])
world_size = int(os.environ["WORLD_SIZE"])
assert world_size <= 8, "world size must be less than or equal to 8"
os.environ["TORCH_FR_DUMP_TEMP_FILE"] = "/tmp/trace_"
os.environ["TORCH_NCCL_DUMP_ON_TIMEOUT"] = "1"
os.environ["TORCH_FR_BUFFER_SIZE"] = "2000"
device = torch.device(f"cuda:{local_rank}")
print(f"{local_rank=} {world_size=} master addr: {os.environ['MASTER_ADDR']} master port: {os.environ['MASTER_PORT']} {device=}")

# Initialize the process group with a small timeout so that jobs fail quickly
dist.init_process_group("nccl", world_size=world_size, rank=local_rank, timeout=timedelta(seconds=1))

a = torch.full((3, 4), float(local_rank), device=device)
# Write some collectives to populate Flight Recorder data
for i in range(2):
  print(f"calling allreduce on {local_rank=}")
  f = dist.all_reduce(a)

# rank0 is doing an additional collective
if local_rank == 0:
  print("rank0 is doing an allreduce on tensor b, but other ranks forgot")
  b = torch.full((4,5), float(local_rank), device=device)
  f = dist.all_reduce(b)

for i in range(2):
  print(f"calling allreduce on {local_rank=}")
  f = dist.all_reduce(a)

torch.cuda.synchronize(device=device)
print(f"{local_rank=} exiting")
