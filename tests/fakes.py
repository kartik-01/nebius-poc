from types import SimpleNamespace

import torch
from torch import nn

VOCAB_SIZE = 128


class FakeTokenizer:
    """Character-level stand-in with a scripted chat template.

    Deterministic and offline, so the batching and boundary logic can be exercised
    without pulling a real tokenizer from the Hub. Multi-character labels tokenize
    into multiple ids, which is how the multi-token candidate path gets covered.
    """

    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [self.token_id(character) for character in text]}

    @staticmethod
    def token_id(character: str) -> int:
        return 1 + (ord(character) % (VOCAB_SIZE - 1))

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        if tokenize:
            raise NotImplementedError("tests only use the string form")
        body = "".join(f"<{message['role']}>{message['content']}" for message in messages)
        return body + "<assistant>" if add_generation_prompt else body


class TinyCausalLM(nn.Module):
    """Shape and gradient harness, not a language model.

    There is no attention, so it says nothing about modeling quality. It exists so
    batching, masking, reshaping, and the backward pass can be checked on CPU in
    milliseconds.
    """

    def __init__(self, vocab_size: int = VOCAB_SIZE, hidden: int = 16, seed: int = 0):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.embed = nn.Embedding(vocab_size, hidden)
        self.head = nn.Linear(hidden, vocab_size)
        with torch.no_grad():
            self.embed.weight.normal_(generator=generator)
            self.head.weight.normal_(generator=generator)
            self.head.bias.zero_()

    def forward(self, input_ids, attention_mask=None):
        return SimpleNamespace(logits=self.head(self.embed(input_ids)))
