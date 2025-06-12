import torch
import torch.distributed as dist

class CrossNodeEventTimer:
    def __init__(self, local_rank, world_size, world_rank):
        self.local_rank = local_rank
        self.world_size = world_size
        self.world_rank = world_rank
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)

        self.records = {f"{world_rank}_p": [], f"{world_rank}_d": []}
        self.buff = []
        self.temp = 0

        self.out_p = None
        self.out_d = None

        self.device = torch.device(f"cuda:{local_rank}")

    def record_start(self, device: torch.device = None):
        device = self.device if device == None else device
        self.start_event.record(torch.cuda.current_stream(device))
    
    def record_end(self, device: torch.device = None):
        device = self.device if device == None else device
        self.end_event.record(torch.cuda.current_stream(device))
        # torch.cuda.synchronize(device)

    def acc_elapsed_time(self, need_synchronize=True) -> float:
        '''return accumulated elapsed time'''
        if need_synchronize: 
            torch.cuda.synchronize(self.device)
            
        duration = self.start_event.elapsed_time(self.end_event)
        self.temp += duration

        return self.temp
    
    def record_elapsed_time(self):
        '''append accumulated elapsed time to buffer'''
        self.buff.append(self.temp)
        self.temp = 0

        return self.buff
    
    def flush_buffer(self, isPrefill=False):
        '''append buffer to records'''
        self.records[f"{self.world_rank}_p" if isPrefill else f"{self.world_rank}_d"].append(self.buff)
        self.buff = []

    def reset(self):
        self.buff = []
        self.temp = 0
        self.records = {f"{self.world_rank}_p": [], f"{self.world_rank}_d": []}

    def all_gather(self, num_batches, max_tokens, group=None):
        '''
        gather from all devices to obtain duration of prefill(all tokens) and decode(token-wise) computation\n
        return with shape of (#devices, #batches) for prefill (out_p)\n
        return with shape of (#devices, #batches, #tokens) for decode (out_d)
        '''
        # print("prefill",torch.tensor(self.records[f"{self.world_rank}_p"]).shape)
        # print(len(self.records[f"{self.world_rank}_d"]))
        # for i in self.records[f"{self.world_rank}_d"]:
        #     print(len(i))
        # print("decode",self.records[f"{self.world_rank}_d"])
        # print("decode",torch.tensor(self.records[f"{self.world_rank}_d"]).shape)

        self.out_p = torch.zeros((self.world_size, num_batches), device=self.device)
        self.out_d = torch.zeros((self.world_size, num_batches, max_tokens), device=self.device)

        dist.all_gather_into_tensor(self.out_p, torch.tensor(self.records[f"{self.world_rank}_p"], device=self.device), group)
        dist.all_gather_into_tensor(self.out_d, torch.tensor(self.records[f"{self.world_rank}_d"], device=self.device), group)

        return self.out_p, self.out_d

    def get_sync_latency(self):
        """
        Example usage of torch.amax and torch.amin for element-wise comparison.

        Example:
            >>> import torch
            >>> d = [
            ...     [
            ...         [5, 10],
            ...         [3, 15]
            ...     ],
            ...     [
            ...         [2, 4],
            ...         [6, 8]
            ...     ]
            ... ]
            >>> tensor_d = torch.tensor(d)

        Compute element-wise max along dim=0
            >>> torch.amax(tensor_d, dim=0)
            tensor([
                [5, 10],  # Max of [5,2] and [10,4]
                [6, 15]   # Max of [3,6] and [15,8]
            ])

        Compute element-wise min along dim=0
            >>> torch.amin(tensor_d, dim=0)
            tensor([
                [2, 4],  # Min of [5,2] and [10,4]
                [3, 8]   # Min of [3,6] and [15,8]
            ])
        """

        max_p = torch.amax(self.out_p, dim=0)
        min_p = torch.amin(self.out_p, dim=0)
        self.out_p = max_p - min_p

        max_d = torch.amax(self.out_d, dim=0)
        min_d = torch.amin(self.out_d, dim=0)
        self.out_d = max_d - min_d

        # print(self.out_p, self.out_d, self.out_d.dtype)
        print("out_p",self.out_p)
        print("out_d",self.out_d)

        print("=" * 20)
        print("average bubble-caused sync latency (ms) per batch")
        print("Prefill", self.out_p.tolist())
        print("Decode", self.out_d.mean(dim=1, keepdim=True).tolist())

        print("=" * 20)
        print("average bubble-caused sync latency (ms) per batch when generating 40 tokens")
        # print("Prefill", self.out_p.sum().item()) // wrong
        # print("Decode", self.out_d.mean(dim=1, keepdim=True).sum().item()) // wrong
        print("Prefill", self.out_p.mean().item())
        print("Decode", self.out_d.sum(dim=1).mean().item())
        print("Sum: ", self.out_p.mean().item()+self.out_d.sum(dim=1).mean().item())

        print("=" * 20)
        print(f"average bubble-caused sync latency (ms) per token for #batches = {self.out_d.shape[0]}")
        print("Prefill", self.out_p.mean().item())
        print("Decode:", self.out_d.mean(dim=1).mean().item())

    def get_mixtral_sync_latency(self):
        pass


class CrossNodeEventTimerV2:
    def __init__(self, local_rank, world_size, world_rank):
        self.local_rank = local_rank
        self.world_size = world_size
        self.world_rank = world_rank
        self.device = torch.device(f"cuda:{local_rank}")

        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)

        self.records = {f"{world_rank}_p": [], f"{world_rank}_d": []}
        self.buff = []
        self.temp = 0

        self.out_p = None
        self.out_d = None

    def record_start(self, device: torch.device = None):
        device = self.device if device == None else device
        self.start_event.record(torch.cuda.current_stream(device))
    
    def record_end(self, device: torch.device = None):
        device = self.device if device == None else device
        self.end_event.record(torch.cuda.current_stream(device))

    def acc_elapsed_time(self, need_synchronize=True) -> float:
        '''return accumulated elapsed time'''
        if need_synchronize: 
            torch.cuda.synchronize(self.device)
            
        duration = self.start_event.elapsed_time(self.end_event)
        self.temp += duration

        return self.temp
    
    def record_elapsed_time(self):
        '''append accumulated elapsed time to buffer'''
        self.buff.append(self.temp)
        self.temp = 0

        return self.buff
    
    def flush_buffer(self, isPrefill=False):
        '''append buffer to records'''
        self.records[f"{self.world_rank}_p" if isPrefill else f"{self.world_rank}_d"].append(self.buff)
        self.buff = []

    def reset(self):
        self.buff = []
        self.temp = 0
        self.records = {f"{self.world_rank}_p": [], f"{self.world_rank}_d": []}

    def all_gather(self, num_batches, max_tokens, num_layers, group=None):
        '''
        gather from all devices to obtain duration of prefill(all tokens) and decode(token-wise) computation\n
        return with shape of (#devices, #batches, 1, #layers) for prefill (out_p)\n
        return with shape of (#devices, #batches, #tokens, #layers) for decode (out_d)
        '''
        # for mixtral, there are two communications for each layer, so we need to multiply num_layers by 2
        self.out_p = torch.zeros((self.world_size, num_batches, 1, num_layers), device=self.device)
        self.out_d = torch.zeros((self.world_size, num_batches, max_tokens-1, num_layers), device=self.device)

        assert self.out_p.shape[1:] == torch.tensor(self.records[f"{self.world_rank}_p"]).reshape((num_batches, -1, num_layers)).shape, f'Shape mismatch for prefill records, expect {self.out_p.shape[1:]} but got {torch.tensor(self.records[f"{self.world_rank}_p"]).reshape((num_batches, -1, num_layers)).shape}'
        assert self.out_d.shape[1:] == torch.tensor(self.records[f"{self.world_rank}_d"]).reshape((num_batches, -1, num_layers)).shape, f"Shape mismatch for decode records, expect {self.out_d.shape[1:]} but got {torch.tensor(self.records[f'{self.world_rank}_d']).reshape((num_batches, -1, num_layers)).shape}"

        dist.all_gather_into_tensor(self.out_p, torch.tensor(self.records[f"{self.world_rank}_p"], device=self.device).reshape((num_batches, -1, num_layers)), group)
        dist.all_gather_into_tensor(self.out_d, torch.tensor(self.records[f"{self.world_rank}_d"], device=self.device).reshape((num_batches, -1, num_layers)), group)

        return self.out_p, self.out_d
    
    def get_mixtral_sync_latency(self):
        """
        this is tested for mixtral
        """

        # MAX - MIN across all devices
        max_p = torch.amax(self.out_p, dim=0)
        min_p = torch.amin(self.out_p, dim=0)
        self.out_p = max_p - min_p

        max_d = torch.amax(self.out_d, dim=0)
        min_d = torch.amin(self.out_d, dim=0)
        self.out_d = max_d - min_d

        # Merge sync latencies in each layer (since each layer has two communications)
        '''
            a = torch.tensor([
                [[1, 2, 3, 4], [2, 2, 3, 4]],
                [[7, 7, 8, 9], [8, 8, 9, 8]],
            ])

            a.reshape(*(a.shape[:, -1]), -1, 2).sum(-1)
            >>> tensor([[[3, 7],[4, 7]], [[14, 17], [16, 17]]])
        '''
        self.out_p = self.out_p.reshape(*(self.out_p.shape[:, -1]), -1, 2).sum(-1)
        self.out_d = self.out_d.reshape(*(self.out_d.shape[:, -1]), -1, 2).sum(-1)

        # Average sync latencies across all layers, then all tokens, then all batches, and finally all devices
        '''
            # x:  (D devices, B batches, T tokens, L layers)
            # shape = (D, B, T, L)

            x1 = x.mean(dim=-1, keepdim=False)      # ⟶ (D, B, T)      ← average over layers
            x2 = x1.mean(dim=-1, keepdim=False)     # ⟶ (D, B)         ← average over tokens
            x3 = x2.mean(dim=-1, keepdim=False)     # ⟶ (D,)           ← average over batches
            result = x3.mean(dim=0,  keepdim=False) # ⟶ scalar         ← average over devices

            # but if you only want the final result, it is equivalent to:
            result = x.mean()
        '''
        self.out_p = self.out_p.mean()
        self.out_d = self.out_d.mean()

        print("Sync Latency Per Layer (ms)")
        print("Prefill: ", self.out_p.item())
        print("Decode: ", self.out_d.item())