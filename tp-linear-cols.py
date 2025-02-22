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

        self.local_in_features = in_features
        self.local_out_features = out_features // self.world_size

        self.linear = nn.Linear(self.local_in_features, self.local_out_features)
    
    def forward(self, x, batch_size):

        input_tensor = torch.zeros(batch_size, self.local_in_features, device=self.device)
        if self.rank == 0:
            dist.scatter(input_tensor, [x for _ in range(self.world_size)], src=0)
        else:
            dist.scatter(input_tensor, None, src=0)

        print(f"input_tensor rank({self.rank}): {input_tensor.shape}")
        print(f"linear.weight rank({self.rank}): {self.linear.weight.shape}")

        # Compute linear transformation without bias
        local_output = self.linear(input_tensor)

        if self.rank == 0:
            output = [torch.zeros(batch_size, self.local_out_features, device=self.device) for _ in range(self.world_size)]
            dist.gather(local_output, output, dst=0)
            output = torch.cat(output, dim=1)
        else:
            dist.gather(local_output, None, dst=0)

        if self.rank == 0:
            print("gather: ", output.shape)
            return output
        else:
            return None
    
def main():
    world_size = torch.cuda.device_count()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = f'cuda:{local_rank}'
    dist.init_process_group(backend='nccl')

    model = Linear(100, 64).to(device)
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
