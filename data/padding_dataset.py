import torch

from torch.utils.data import Dataset
from transformers import TokenizersBackend
from datasets import Dataset as HF_Dataset


class PretrainPaddingDataset(Dataset):
    def __init__(self, tokenizer: TokenizersBackend, max_seq: int, dataset: HF_Dataset):
        self.tokenizer = tokenizer
        self.max_seq = max_seq
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(
        self,
        index: int,
        col_name: str = "text",
        add_bos_id: bool = False,
        add_eos_id: bool = True,
    ) -> None:
        sample = self.dataset[index]
        seq_len = self.max_seq + 1
        if add_bos_id:
            seq_len -= 1
        if add_eos_id:
            seq_len -= 1
        ids: list[int] = self.tokenizer(
            str(sample[col_name]),
            add_special_tokens=False,
            max_length=seq_len,
            truncation=True,
        ).input_ids

        if add_bos_id:
            ids.insert(0, self.tokenizer.bos_token_id)
        if add_eos_id:
            ids.append(self.tokenizer.eos_token_id)
        if hasattr(self.tokenizer, "pod_token_id"):
            pad_token_id = self.tokenizer.pod_token_id
        else:
            pad_token_id = -100
        if len(ids) < self.max_seq + 1:
            ids = ids + [pad_token_id] * (self.max_seq + 1 - len(ids))

        ids_tensor = torch.tensor(ids, dtype=torch.long)
        x = ids_tensor[:-1]
        y = ids_tensor[1:].clone()
        y[y == pad_token_id] = -100
        return x, y


if __name__ == "__main__":
    from transformers import AutoTokenizer
    from datasets import load_dataset

    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

    dataset = load_dataset("roneneldan/TinyStories", "default", split="train")

    dataset = PretrainPaddingDataset(tokenizer, max_seq=1024, dataset=dataset)
    import torch.utils.data

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    for x, y in dataloader:
        assert x.shape == (1, 1024), f"x.shape: {x.shape}"
        assert y.shape == (1, 1024), f"y.shape: {y.shape}"
        break
