import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        self.PAD_ID = 0
        self.UNK_ID = 1
        self.BOS_ID = 2
        self.EOS_ID = 3
        
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    
    def build_vocab(self, texts: List[str]) -> None:
        self.word_to_id = {
            self.pad_token: self.PAD_ID,
            self.unk_token: self.UNK_ID,
            self.bos_token: self.BOS_ID,
            self.eos_token: self.EOS_ID
        }
        
        unique_words = set()
        for text in texts:
            words = text.split()
            for word in words:
                unique_words.add(word)

        sorted_words = sorted(list(unique_words))
        
        curr_id = 4
        for word in sorted_words:
            if word not in self.word_to_id:
                self.word_to_id[word] = curr_id
                curr_id += 1
                
        self.id_to_word = {v: k for k, v in self.word_to_id.items()}
        self.vocab_size = len(self.word_to_id)
    
    def encode(self, text: str) -> List[int]:
        words = text.split()
        token_ids = []
        
        for word in words:
            token_id = self.word_to_id.get(word, self.UNK_ID)
            token_ids.append(token_id)
            
        return token_ids
    
    def decode(self, ids: List[int]) -> str:
        words = []
        for tid in ids:
            word = self.id_to_word.get(tid, self.unk_token)
            words.append(word)
            
        return " ".join(words)