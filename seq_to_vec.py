import re

import numpy as np
import torch
import gc
from transformers import AutoModel, BertModel, BertTokenizer, AutoTokenizer
from tqdm import tqdm
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model_to_dim = {
    "ProtBert": 1024,
    "ChemBERTa": 768,
    "MoLFormer": 768,
    "esm3": 1152
}


class Esm3Embedder:
    def __init__(self):
        from esm.models.esmc import ESMC
        from esm.sdk.api import ESMProtein, LogitsConfig
        self.ESMProtein = ESMProtein
        self.LogitsConfig = LogitsConfig
        self.model = ESMC.from_pretrained("esmc_600m", device=device).eval()

    def to_vec(self, seq_list: str):
        res = []
        for seq in seq_list:
            if len(seq) > 1023:
                seq = seq[:1023]
            try:
                protein = self.ESMProtein(sequence=seq)
                protein = self.model.encode(protein)
                conf = self.LogitsConfig(return_embeddings=True, sequence=True)
                vec = self.model.logits(protein, conf).embeddings[0]
                res.append(vec.mean(dim=0).cpu().numpy().flatten())
            except Exception as e:
                print(e)
                res.append(np.zeros(model_to_dim["esm3"]))
        return np.array(res)


class PortBert:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
        self.model = BertModel.from_pretrained("Rostlab/prot_bert").to(device).eval()

    def to_vec(self, seq_list: str):
        seq_list = [" ".join(list(re.sub(r"[UZOB]", "X", seq))) for seq in seq_list]
        inputs = self.tokenizer(seq_list, return_tensors='pt', padding="longest", truncation=True, max_length=1024)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            embedding_repr = self.model(**inputs)
        vec = embedding_repr.last_hidden_state.mean(dim=1)
        return vec.detach().cpu().numpy()


class MoLFormer:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("ibm/MoLFormer-XL-both-10pct", trust_remote_code=True)
        self.model = AutoModel.from_pretrained("ibm/MoLFormer-XL-both-10pct", deterministic_eval=True,
                                               trust_remote_code=True).to(device).eval()

    def to_vec(self, seq_list: str):
        inputs = self.tokenizer(seq_list, return_tensors='pt', padding="longest", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        vec = outputs.pooler_output
        return vec.detach().cpu().numpy()


class ChemBERTa:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
        self.model = AutoModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1").to(device).eval()

    def to_vec(self, seq_list: str):
        inputs = self.tokenizer(seq_list, return_tensors='pt', padding="longest", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            hidden_states = self.model(**inputs)[0]
        vec = torch.mean(hidden_states, dim=1)
        vec_cpu = vec.detach().cpu().numpy()
        del hidden_states, vec
        for k in inputs:
            del inputs[k]
        del inputs
        torch.cuda.empty_cache()

        return vec_cpu


def get_model(model):
    if model == "ProtBert":
        return PortBert()
    elif model == "ChemBERTa":
        return ChemBERTa()
    elif model == "MoLFormer":
        return MoLFormer()
    elif model == "esm3":
        return Esm3Embedder()
    else:
        raise ValueError("Model not found")


def apply_model_in_batches(model, seq_list, batch_size):
    vecs = []
    for i in tqdm(range(0, len(seq_list), batch_size)):
        end_index = min(i + batch_size, len(seq_list))
        vecs.append(model.to_vec(seq_list[i:i + end_index]))
        torch.cuda.empty_cache()
        gc.collect()
    return np.concatenate(vecs)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Convert sequence to vector')
    parser.add_argument('--model', type=str, help='Model to use', default="ChemBERTa",
                        choices=["ProtBert", "ChemBERTa", "MoLFormer", "esm3"])
    parser.add_argument('--batch_size', type=int, help='Batch size', default=32)
    args = parser.parse_args()
    model_name = args.model
    model = get_model(model_name)
    input_file = "data/protein.fasta" if model_name in ["ProtBert", "esm3"] else "data/ligands.smi"
    output_file = f"data/{model_name}_vec.npy"
    with open(input_file, 'r') as f:
        seq_list = f.read().splitlines()
    vec = apply_model_in_batches(model, seq_list, args.batch_size)
    np.save(output_file, vec)
