from torch.utils.data import Dataset
import os
import numpy as np
from .index_dataset import IndexDataset
import math
import torch

class SimpleMegatronDataset(Dataset):
    """
    用于预训练任务的简单数据集封装类。
    
    该类虽然名为"Simple"，但内部复现了 Megatron-LM 的核心数据索引构建算法 (Dataset Index Builder)，
    主要用于处理大规模预训练数据的流式读取。它不支持多数据集混合(Blending)和训练/验证集划分，
    但在单一数据源的处理上具备企业级的高效性和随机性。

    核心算法与实现原理：

    1. **内存映射 (Memory Mapping)**:
       底层依赖 `IndexDataset` 读取 `.bin` 和 `.idx` 文件。通过 `numpy.memmap` 实现大文件的
       虚拟内存映射，使得即使处理 TB 级数据，物理内存占用也仅限于当前 Batch 的数据量，
       完全避免 OOM (Out Of Memory)。

    2. **两级索引机制 (Two-Level Indexing)**:
       为了同时实现"跨文档拼接"和"随机读取"，本类构建了两个关键的索引结构：
       - **`doc_indices` (文档索引)**: 这是一个打乱后的文档 ID 列表。如果训练需要覆盖 D 个文档 E 个 Epoch，
         该列表长度约为 D * E。我们在构建时会针对每个 Epoch 独立打乱文档顺序（基于 `seed + epoch`），
         既保证了随机性，又保证了磁盘读取的连续性（减少随机 IO）。
       - **`sample_indices` (样本索引)**: 这是一个元组列表 `[(doc_list_idx, offset), ...]`, 
         记录了每个训练样本的"逻辑起始位置"。
         - `doc_list_idx`: 指向 `doc_indices` 中的下标，表示该样本从哪个文档开始。
         - `offset`: 表示该文档内的起始 Token 偏移量。

    3. **贪婪拼接策略 (Greedy Stitching)**:
       在 `build_dataset_indices` 中，使用一个贪婪算法遍历 `doc_indices`：
       - 每次尝试“吃掉” `seq_len + 1` 个 Token 作为一个样本。
       - 如果当前文档剩余 Token 不足，则无缝跳转到下一个文档继续读取，直到凑齐。
       - 这种策略确保了 Token 的利用率接近 100%，没有 Padding 浪费。

    4. **双重随机 (Double Shuffling)**:
       - **第一重**: `doc_indices` 的 Epoch 级随机打乱。
       - **第二重**: 生成 `sample_indices` 后，再次生成一个 `shuffle_indices` 数组对其进行
         全局随机映射。这确保了训练过程中样本的出现顺序是完全随机的。

    5. **索引缓存 (Index Caching)**:
       - 自动将构建好的索引（doc, sample, shuffle）保存为 `.npy` 文件。
       - 文件名包含 seed、seq_len 等关键参数，确保参数变更时自动失效。
       - 实现秒级启动和断点恢复时的严格数据一致性。

    6. **严格 Token 限制 (Strict Token Limit)**:
       - 在构建样本时，严格遵守 `total_token` 限制。
       - 即使预先准备了多余的文档索引，一旦生成的样本总 Token 数达到 `total_token`，
         构建过程立即停止，确保训练步数精准符合预期。

    Args:
        idx_path (str): 数据集索引文件路径 (.idx)。
        bin_path (str): 数据集二进制文件路径 (.bin)。
        seq_len (int): 序列长度 (Sequence Length)。实际每个样本会读取 seq_len + 1 个 Token (Input + Label)。
        total_token (int): 本次训练预期的总 Token 数。类会自动计算需要的 Epoch 数并构建足够的索引。
        seed (int): 随机种子，用于确保数据打乱的可复现性。
    """
    def __init__(self, idx_path:str,bin_path:str,seq_len:int=1024,total_token:int=0, seed:int=42):
        ...
        if not os.path.exists(idx_path):
            raise FileNotFoundError(f"Index file {idx_path} not found.")
        if not os.path.exists(bin_path):
            raise FileNotFoundError(f"Bin file {bin_path} not found.")
        if total_token <= seq_len:
            raise ValueError(f"total_token must be greater than seq_len: {seq_len}")
        
        self.seq_len = seq_len
        self.total_token = total_token
        self.seed = seed
        self.dataset: IndexDataset = IndexDataset(idx_path,bin_path,seq_len=seq_len)
        self.build_dataset_indices()
        
        
    def build_dataset_indices(self):
        # 1. 尝试生成缓存路径
        # 缓存文件名包含关键参数的 hash 或标识，确保参数变了会自动重新生成
        cache_dir = os.path.dirname(self.dataset.get_bin_path())
        idx_name = os.path.basename(self.dataset.get_bin_path()).replace('.bin', '')
        cache_base_name = f"{idx_name}_seq{self.seq_len}_seed{self.seed}_tokens{self.total_token}"
        
        path_doc_idx = os.path.join(cache_dir, f"{cache_base_name}_doc_idx.npy")
        path_sample_idx = os.path.join(cache_dir, f"{cache_base_name}_sample_idx.npy")
        path_shuffle_idx = os.path.join(cache_dir, f"{cache_base_name}_shuffle_idx.npy")
        
        # 2. 检查缓存是否存在
        if os.path.exists(path_doc_idx) and os.path.exists(path_sample_idx) and os.path.exists(path_shuffle_idx):
            print(f"Loading dataset indices from cache: {cache_base_name} ...")
            self.doc_indices = np.load(path_doc_idx, mmap_mode='r')
            self.sample_indices = np.load(path_sample_idx, mmap_mode='r')
            self.shuffle_indices = np.load(path_shuffle_idx, mmap_mode='r')
            print(f"Loaded {len(self.sample_indices)} samples.")
            return

        print("Building dataset indices (cache miss)...")
        
        # 需要重复文档，直到总 token 数达到 total_token
        total_epoch = math.ceil(self.total_token / self.dataset.get_total_tokens())
        
        self.doc_indices = []
        # 填充文档索引，并在每个 epoch 内打乱
        total_docs = self.dataset.get_total_docs()
        for i in range(total_epoch):
            epoch_doc_indices = np.arange(total_docs)
            # 使用 seed + epoch 确保每个 epoch 的随机性是确定的但又不同
            rng = np.random.RandomState(self.seed + i)
            rng.shuffle(epoch_doc_indices)
            self.doc_indices.extend(epoch_doc_indices.tolist())
        
        
        # 4. 保存 sample_indices
        self.sample_indices = []
        doc_idx_in_list = 0
        offset_in_doc = 0
        
        # 目标是构建足够的 sample
        # 每个 sample 需要 seq_len + 1 个 token (input + label)
        block_size = self.seq_len + 1
        
        # 追踪已生成的 token 总量
        generated_tokens = 0 
        
        while True:
            # 【新增逻辑】: 检查是否已经满足 total_token 需求
            # 注意: 这里用 generated_tokens + block_size 来预判
            if generated_tokens + block_size > self.total_token:
                break
            
            # 记录当前样本起始位置 (doc_index_in_list, offset_in_doc)
            self.sample_indices.append((doc_idx_in_list, offset_in_doc))
            generated_tokens += block_size
            
            # 计算是否已经满足 total_token 需求 (近似)
            # 或者当我们的 doc_indices 耗尽时停止
            # 这里简单判定：如果 doc_idx_in_list 超出了范围，就停止
            if doc_idx_in_list >= len(self.doc_indices):
                self.sample_indices.pop() # 最后一个非完整
                break
                
            remaining = block_size
            
            while remaining > 0:            
                doc_id = self.doc_indices[doc_idx_in_list]
                doc_len = self.dataset.get_doc_len(doc_id)
                available = doc_len - offset_in_doc
                
                if available >= remaining:
                    # 当前文档足够满足当前样本需求
                    offset_in_doc += remaining
                    remaining = 0
                    # 准备下一个样本
                else:
                    # 不够满足当前样本需求，用完当前文档，跳到下一个文档
                    remaining -= available
                    doc_idx_in_list += 1
                    offset_in_doc = 0
            
        print(f"Built {len(self.sample_indices)} samples.")
        
        # 3. 生成 shuffle_indices
        self.shuffle_indices = np.arange(len(self.sample_indices))
        # 使用基于 seed 的随机状态进行打乱，确保多卡环境下 shuffle 结果一致
        rng = np.random.RandomState(self.seed)
        rng.shuffle(self.shuffle_indices)
        
        # 4. 保存缓存
        print(f"Saving dataset indices to cache: {cache_base_name} ...")
        # 转换为 numpy array 保存
        np.save(path_doc_idx, np.array(self.doc_indices, dtype=np.int32))
        np.save(path_sample_idx, np.array(self.sample_indices, dtype=np.int32))
        np.save(path_shuffle_idx, np.array(self.shuffle_indices, dtype=np.int64))
        print("Cache saved.")
        
    def __len__(self):
        return len(self.sample_indices)
    
    
    def __getitem__(self, idx):        
        # 0. 映射到随机打乱后的 idx
        
        idx = self.shuffle_indices[idx]
        
        # 1. 获取起始位置
        doc_list_idx, offset = self.sample_indices[idx]
        
        # 2. 开始拼接 token
        tokens_list = []
        remaining = self.seq_len + 1
        
        while remaining > 0:                
            doc_id = self.doc_indices[doc_list_idx]
            doc_len = self.dataset.get_doc_len(doc_id)
            available = doc_len - offset
            
            # 决定读取多少
            to_read = min(available, remaining)
            
            # 读取片段
            chunk = self.dataset.get(doc_id, offset=offset, length=to_read)
            tokens_list.append(chunk)
            
            remaining -= to_read
            
            if remaining > 0:
                # 跨越文档
                doc_list_idx += 1
                offset = 0
                
        full_seq = np.concatenate(tokens_list)
    
        
        x = torch.from_numpy(full_seq[:-1].astype(np.int64))
        y = torch.from_numpy(full_seq[1:].astype(np.int64))
        
        return x, y
    
    
if __name__ == "__main__":
    
    dataset = SimpleMegatronDataset(idx_path="train_data/tiny_stories.idx", bin_path="train_data/tiny_stories.bin", seq_len=1024, total_token=100 * 10000, seed=42)
    import torch.utils.data
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False)
    for x, y in dataloader:
        assert x.shape == (16, 1024), f"x.shape: {x.shape}"
        assert y.shape == (16, 1024), f"y.shape: {y.shape}"
        break