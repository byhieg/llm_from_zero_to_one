import logging

from dataset.padding_dataset import PretrainPaddingDataset


class DummyEncoding:
    def __init__(self, input_ids):
        self.input_ids = input_ids


class DummyTokenizer:
    bos_token_id = 101
    eos_token_id = 102
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False, max_length=None, truncation=True):
        tokens = [11, 12]
        if max_length is not None:
            tokens = tokens[:max_length]
        return DummyEncoding(tokens)

    def decode(self, token_ids, skip_special_tokens=False):
        if any(token_id < 0 for token_id in token_ids):
            raise OverflowError("negative token id")
        return ",".join(str(token_id) for token_id in token_ids)


def test_padding_dataset_keeps_padding_out_of_x():
    dataset = PretrainPaddingDataset(
        tokenizer=DummyTokenizer(),
        max_seq=4,
        dataset=[{"text": "hello"}],
        dataset_config={"col_name": "text", "add_bos_id": False, "add_eos_id": True},
    )

    x, y = dataset[0]

    assert x.tolist() == [11, 12, 102, 0]
    assert y.tolist() == [12, 102, -100, -100]


def test_padding_dataset_print_sample_hides_right_padding(caplog):
    dataset = PretrainPaddingDataset(
        tokenizer=DummyTokenizer(),
        max_seq=4,
        dataset=[{"text": "hello"}],
        dataset_config={"col_name": "text", "add_bos_id": False, "add_eos_id": True},
    )

    with caplog.at_level(logging.INFO):
        dataset.print_sample()

    assert "x: 11,12,102" in caplog.text
    assert "y (non-padding): 12,102" in caplog.text
