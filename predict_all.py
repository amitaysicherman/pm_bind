from seq_to_vec import get_model

from train_main import BindingModel, model_to_dim
import torch
import numpy as np
import os
from safetensors.torch import load_file

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_last_cp(base_dir):
    files = os.listdir(base_dir)
    cp_files = [f for f in files if "checkpoint" in f]
    cp_files = sorted(cp_files)
    return cp_files[-1]


def get_affinity_model(protein_model, molecule_model):
    protein_dim = model_to_dim[protein_model]
    molecule_dim = model_to_dim[molecule_model]
    model = BindingModel(protein_dim, molecule_dim)

    cp_base = f"results_{protein_model}_{molecule_model}_1024_0.0001"
    cp_path = os.path.join(cp_base, get_last_cp(cp_base), "model.safetensors")

    # Load the safetensors file
    state_dict = load_file(cp_path)
    model.load_state_dict(state_dict)
    model.to(device)
    return model
def get_scores(protein_seq, molecules_seqs, protein_model, molecule_model, affinity_model):
    protein_vec = models_dict[protein_model].to_vec([protein_seq])
    protein_vec = torch.Tensor(protein_vec).to(device)

    molecule_vec = models_dict[molecule_model].to_vec(molecules_seqs)
    molecule_vec = torch.Tensor(molecule_vec).to(device)

    # repeat protein_vec for each molecule
    protein_vec = protein_vec.repeat(len(molecule_vec), 1)
    res = affinity_model(protein_vec, molecule_vec)
    print(res)


protein_seq = "MAGKVIKCKAAVLW"
molecules_seqs = ["CCO", "CCC", "CCCC", "CCCCC"]
protein_models = ["ProtBert", "esm3"]
molecule_models = ["MoLFormer", "ChemBERTa"]
models_dict = {}
for protein_model in protein_models:
    models_dict[protein_model] = get_model(protein_model)

for molecule_model in molecule_models:
    models_dict[molecule_model] = get_model(molecule_model)

model_pairs = [(protein_model, molecule_model) for protein_model in protein_models for molecule_model in
               molecule_models]
for protein_model, molecule_model in model_pairs:
    affinity_model = get_affinity_model(protein_model, molecule_model)
    get_scores(protein_seq, molecules_seqs, protein_model, molecule_model, affinity_model)
