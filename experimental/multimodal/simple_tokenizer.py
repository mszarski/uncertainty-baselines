# coding=utf-8
# Copyright 2022 The Uncertainty Baselines Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Implementation of CLIP's tokenizer.

Forked from third_party/py/clip/simple_tokenizer.py.
"""
import functools
import gzip
import html
from typing import Callable, Optional
from etils import epath
import numpy as np
import regex as re
import tensorflow as tf


# pylint: disable=line-too-long
@functools.lru_cache()
def default_bpe():
  path = epath.resource_path('uncertainty_baselines') / 'experimental/multimodal/bpe_simple_vocab_16e6.txt.gz'
  return path
# pylint: enable=line-too-long


@functools.lru_cache()
def bytes_to_unicode():
  """Returns list of utf-8 byte and a corresponding list of unicode strings.

  The reversible bpe codes work on unicode strings. This means you need a large
  # of unicode characters in your vocab if you want to avoid UNKs. When you're
  at something like a 10B token dataset you end up needing around 5K for decent
  coverage. This is a signficant percentage of your normal, say, 32K bpe vocab.
  To avoid that, we want lookup tables between utf-8 bytes and unicode strings.

  And avoids mapping to whitespace/control characters the bpe code barfs on.
  """
  bs = list(range(ord('!'),
                  ord('~') + 1)) + list(range(ord('¡'),
                                              ord('¬') + 1)) + list(
                                                  range(ord('®'),
                                                        ord('ÿ') + 1))
  cs = bs[:]
  n = 0
  for b in range(2**8):
    if b not in bs:
      bs.append(b)
      cs.append(2**8 + n)
      n += 1
  cs = [chr(n) for n in cs]
  return dict(zip(bs, cs))


def get_pairs(word):
  """Return set of symbol pairs in a word.

  Arguments:
    word: tuple of symbols (symbols being variable-length strings).

  Returns:
    a `set` of  symbol pairs in `word`.
  """
  pairs = set()
  prev_char = word[0]
  for char in word[1:]:
    pairs.add((prev_char, char))
    prev_char = char
  return pairs


def basic_clean(text):
  text = html.unescape(html.unescape(text))
  return text.strip()


def whitespace_clean(text):
  text = re.sub(r'\s+', ' ', text)
  text = text.strip()
  return text


class SimpleTokenizer(object):
  """CLIP's tokenizer."""

  def __init__(self,
               bpe_path: Optional[str] = None,
               cache_encodings: bool = True):
    self.cache_encodings = cache_encodings
    if bpe_path is None:
      bpe_path = default_bpe()

    self.byte_encoder = bytes_to_unicode()
    self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
    with epath.Path(bpe_path).open('rb') as f:
      merges = gzip.open(f).read().decode('utf-8').split('\n')
    merges = merges[1:49152 - 256 - 2 + 1]
    merges = [tuple(merge.split()) for merge in merges]
    vocab = list(bytes_to_unicode().values())
    vocab = vocab + [v + '</w>' for v in vocab]
    for merge in merges:
      vocab.append(''.join(merge))
    vocab.extend(['<|startoftext|>', '<|endoftext|>'])
    self.encoder = dict(zip(vocab, range(len(vocab))))
    self.decoder = {v: k for k, v in self.encoder.items()}
    self.bpe_ranks = dict(zip(merges, range(len(merges))))
    self.cache = {
        '<|startoftext|>': '<|startoftext|>',
        '<|endoftext|>': '<|endoftext|>'
    }
    self.pat = re.compile(
        r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""",
        re.IGNORECASE)

  def bpe(self, token):
    if self.cache_encodings and token in self.cache:
      return self.cache[token]
    word = tuple(token[:-1]) + (token[-1] + '</w>',)
    pairs = get_pairs(word)

    if not pairs:
      return token + '</w>'

    while True:
      bigram = min(
          pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
      if bigram not in self.bpe_ranks:
        break
      first, second = bigram
      new_word = []
      i = 0
      while i < len(word):
        try:
          j = word.index(first, i)
          new_word.extend(word[i:j])
          i = j
        except:
          new_word.extend(word[i:])
          break

        if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
          new_word.append(first + second)
          i += 2
        else:
          new_word.append(word[i])
          i += 1
      new_word = tuple(new_word)
      word = new_word
      if len(word) == 1:
        break
      else:
        pairs = get_pairs(word)
    word = ' '.join(word)
    if self.cache_encodings:
      self.cache[token] = word
    return word

  def encode(self, text):
    bpe_tokens = []
    text = whitespace_clean(basic_clean(text)).lower()
    for token in re.findall(self.pat, text):
      token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
      bpe_tokens.extend(
          self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
    return bpe_tokens

  def decode(self, tokens):
    text = ''.join([self.decoder[token] for token in tokens])
    text = bytearray([self.byte_decoder[c] for c in text]).decode(
        'utf-8', errors='replace').replace('</w>', ' ')
    return text


def make_tokenize_fn(tokenizer: SimpleTokenizer,
                     max_len: int = 77) -> Callable[[tf.Tensor], np.ndarray]:
  """Creates a function that accepts text and returns BPE tokens."""
  sot_token = tokenizer.encoder['<|startoftext|>']
  eot_token = tokenizer.encoder['<|endoftext|>']

  def tokenize(text: tf.Tensor) -> np.ndarray:
    # Adapted from http://google3/third_party/py/clip/clip.py?l=127
    tokens = [sot_token] + tokenizer.encode(
        text.numpy().decode('utf-8')) + [eot_token]
    result = np.zeros((max_len,), dtype=np.int64)
    result[:len(tokens)] = np.asarray(tokens[:max_len])
    return result

  return tokenize
