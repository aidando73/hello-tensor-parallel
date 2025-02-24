import torch
import torch.nn as nn
import torch.distributed as dist
import os

class Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.device = f'cuda:{self.rank}'

        self.local_in_features = in_features // self.world_size
        self.local_out_features = out_features

        self.linear = nn.Linear(self.local_in_features, self.local_out_features)
    
    def forward(self, x, batch_size):
        
        local_input = torch.zeros(batch_size, self.local_in_features, device=self.device)

        dist.scatter(local_input, list(x.chunk(self.world_size, dim=1)) if self.rank == 0 else None, src=0)

        print(f"local_input rank({self.rank}): {local_input.shape}")
        print(f"linear.weight rank({self.rank}): {self.linear.weight.shape}")

        # Compute linear transformation without bias
        local_output = nn.functional.linear(local_input, self.linear.weight, None)

        dist.reduce(local_output, dst=0, op=dist.ReduceOp.SUM)


        # Add bias only on rank 0 after reduction
        if self.rank == 0:
            local_output = local_output + self.linear.bias

        return local_output
    
def main():
    world_size = torch.cuda.device_count()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = f'cuda:{local_rank}'
    dist.init_process_group(backend='nccl')

    model = Linear(100, 50).to(device)
    batch_size = 32

    if dist.get_rank() == 0:
        input_tensor = torch.randn(batch_size, 100, device=device)
    else:
        input_tensor = None

    output = model(input_tensor, batch_size)
    if dist.get_rank() == 0:
        print(output, output.shape)

if __name__ == "__main__":
    main()
