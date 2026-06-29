import torch

from torch.utils.data import Dataset
from transformers import TokenizersBackend
from datasets import Dataset as HF_Dataset
from logger import get_logger

logger = get_logger(__name__)


class PretrainPaddingDataset(Dataset):
    """训练阶段使用的 padding 数据集。"""

    def __init__(
        self,
        tokenizer: TokenizersBackend,
        max_seq: int,
        dataset: HF_Dataset,
        dataset_config: dict = None,
    ):
        self.tokenizer = tokenizer
        self.max_seq = max_seq
        self.dataset = dataset
        self.col_name = dataset_config.get("col_name", "text")
        self.add_bos_id = dataset_config.get("add_bos_id", False)
        self.add_eos_id = dataset_config.get("add_eos_id", True)
        self._sample_preview_logged = False

    def __len__(self) -> int:
        return len(self.dataset)

    def _trim_right_padding(
        self, token_ids: list[int], pad_token_id: int | None
    ) -> list[int]:
        if pad_token_id is None:
            return token_ids
        valid_token_ids = token_ids[:]
        while valid_token_ids and valid_token_ids[-1] == pad_token_id:
            valid_token_ids.pop()
        return valid_token_ids

    def __getitem__(self, index: int) -> None:
        sample = self.dataset[index]
        seq_len = self.max_seq + 1
        if self.add_bos_id:
            seq_len -= 1
        if self.add_eos_id:
            seq_len -= 1
        ids: list[int] = self.tokenizer(
            str(sample[self.col_name]),
            add_special_tokens=False,
            max_length=seq_len,
            truncation=True,
        ).input_ids

        if self.add_bos_id:
            ids.insert(0, self.tokenizer.bos_token_id)
        if self.add_eos_id:
            ids.append(self.tokenizer.eos_token_id)
        pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(self.tokenizer, "eos_token_id", 0)
        if len(ids) < self.max_seq + 1:
            ids = ids + [pad_token_id] * (self.max_seq + 1 - len(ids))
        if index == 0 and not self._sample_preview_logged:
            logger.info("sample: %s", sample[self.col_name])
            logger.info(
                "ids: %s,bos_token_id:%s,eos_token_id:%s",
                ids,
                self.tokenizer.bos_token_id,
                self.tokenizer.eos_token_id,
            )
            self._sample_preview_logged = True
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
