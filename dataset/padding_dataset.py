import torch

from torch.utils.data import Dataset
from transformers import TokenizersBackend
from datasets import Dataset as HF_Dataset
from logger import get_logger

logger = get_logger(__name__)
class PretrainPaddingDataset(Dataset):
    def __init__(self, tokenizer: TokenizersBackend, max_seq: int, dataset: HF_Dataset, dataset_config: dict = None):
        self.tokenizer = tokenizer
        self.max_seq = max_seq
        self.dataset = dataset
        self.dataset_config = dataset_config or {
            'col_name': 'text',
            'add_bos_id': False,
            'add_eos_id': True,
        }

    def __len__(self) -> int:
        return len(self.dataset)

    def _trim_right_padding(self, token_ids: list[int], pad_token_id: int | None) -> list[int]:
        if pad_token_id is None:
            return token_ids
        valid_token_ids = token_ids[:]
        while valid_token_ids and valid_token_ids[-1] == pad_token_id:
            valid_token_ids.pop()
        return valid_token_ids

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
        pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(self.tokenizer, "eos_token_id", 0)
        if len(ids) < self.max_seq + 1:
            ids = ids + [pad_token_id] * (self.max_seq + 1 - len(ids))
        if index == 0:
            logger.log_once(f"sample: {sample[col_name]}")
            logger.log_once(f"ids: {ids}")
        ids_tensor = torch.tensor(ids, dtype=torch.long)
        x = ids_tensor[:-1]
        y = ids_tensor[1:].clone()
        y[y == pad_token_id] = -100
        return x, y

    def print_sample(self):
        x, y = self.__getitem__(
            0,
            col_name=self.dataset_config['col_name'],
            add_bos_id=self.dataset_config['add_bos_id'],
            add_eos_id=self.dataset_config['add_eos_id'],
        )
        pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
        x_valid = self._trim_right_padding(x.tolist(), pad_token_id)
        y_valid = y[y >= 0].tolist()
        logger.info(f"x: {self.tokenizer.decode(x_valid)}")
        logger.info(f"y (non-padding): {self.tokenizer.decode(y_valid)}")

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
