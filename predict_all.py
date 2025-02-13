import pandas as pd

from seq_to_vec import get_model

from train_main import BindingModel, model_to_dim
import torch
import numpy as np
import os
from safetensors.torch import load_file
from tqdm import tqdm

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

    state_dict = load_file(cp_path)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model


def get_scores(protein_seq, molecules_seqs, protein_model, molecule_model, affinity_model):
    protein_vec = models_dict[protein_model].to_vec([protein_seq])
    protein_vec = torch.Tensor(protein_vec).to(device)

    molecule_vec = models_dict[molecule_model].to_vec(molecules_seqs)
    molecule_vec = torch.Tensor(molecule_vec).to(device)

    # repeat protein_vec for each molecule
    protein_vec = protein_vec.repeat(len(molecule_vec), 1)
    res = affinity_model(protein_vec, molecule_vec)
    res = res.logits.detach().cpu().numpy().flatten().tolist()
    return res


protein_seq = "MCDEDETTALVCDNGSGLVKAGFAGDDAPRAVFPSIVGRPRHQGVMVGMGQKDSYVGDEAQSKRGILTLKYPIEHGIITNWDDMEKIWHHTFYNELRVAPEEHPTLLTEAPLNPKANREKMTQIMFETFNVPAMYVAIQAVLSLYASGRTTGIVLDSGDGVTHNVPIYEGYALPHAIMRLDLAGRDLTDYLMKILTERGYSFVTTAEREIVRDIKEKLCYVALDFENEMATAASSSSLEKSYELPDGQVITIGNERFRCPETLFQPSFIGMESAGIHETTYNSIMKCDIDIRKDLYANNVMSGGTTMYPGIADRMQKEITALAPSTMKIKIIAPPERKYSVWIGGSILASLSTFQQMWITKQEYDEAGPSIVHRKCF"
# molecules_seqs = [
#     r'CC1C(=O)NC2CC3=C(NC4=CC=CC=C34)SCC(C(=O)N5CC(CC5C(=O)N1)O)NC(=O)C(NC(=O)C(NC(=O)C(NC2=O)CC(C)(CO)O)C)C(C)O',
#     r'C[C@@H]\\1C[C@@H](OC(=O)C[C@@H](NC(=O)[C@H](N(C(=O)[C@@H](NC(=O)[C@H](C/C(=C1)/C)C)C)C)CC2=C(NC3=CC=CC=C32)Br)C4=CC=C(C=C4)O)C',
#     r'C[C@H]1C[C@H]([C@H](C[C@H](OC(=O)[C@H](N(C(=O)CNC(=O)[C@H](C1)C)C)CC2=CC(=C(C=C2)O)I)C(C)C)O)C',
#     r'C[C@@H]1CCC[C@H](/C=C/C(=O)O[C@]23[C@@H](/C=C/C1)[C@@H](C(=C)[C@H]([C@H]2[C@@H](NC3=O)CC4=CC=CC=C4)C)O)O',
#     r'C[C@H]1C/C=C/[C@H]2[C@@H](C(=C)[C@H]([C@@H]3[C@@]2([C@@H](/C=C/[C@@](C1=O)(C)O)OC(=O)C)C(=O)N[C@H]3CC4=CC=CC=C4)C)O',
#     r'C[C@H]1C[C@H](C[C@@H](O1)CC[C@@H](C)[C@@H]([C@H](C)[C@@H]2[C@H]([C@@H](C[C@@H]([C@@H]([C@H](C[C@@H]3CC=C[C@H](O3)C[C@@H](C/C=C(\\C=C\\C(=O)O[C@@H]([C@H]([C@@H](C[C@@H]([C@@H]([C@H](C[C@@H]4CC=C[C@H](O4)C[C@H](C/C=C(/C=C/C(=O)O2)\\C)O)OC)C)O)O)C)[C@@H](C)[C@H]([C@@H](C)CC[C@H]5C[C@@H](C[C@@H](O5)C)OC)O)/C)O)OC)C)O)O)C)O)OC',
#     r'C[C@H]/1CC[C@@H]2C[C@H](C[C@@](O2)([C@@H]3CSC(=O)N3)O)OC(=O)/C=C(\\CC/C=C/C=C1)/C',
#     r'C[C@H]1/C=C/[C@H](C[C@@H]([C@H]([C@H]2[C@H](C(=CC(=O)O2)C[C@@H](C/C(=C\\C=C\\C(=O)O[C@@H]([C@H](/C=C/[C@@H](C[C@@H]1O)OC)C)[C@@H](C)[C@H]([C@@H](C)CCC(=O)[C@H](C)[C@@H](C/C=C/N(C)C=O)OC)OC)/C)OC)O)C)OC)OC',
#     r'C[C@@H]1CC[C@@H](/C(=C/C[C@H](C[C@H](/C=C/C[C@H](OC(=O)/C=C/C=C/C[C@H]([C@H]([C@@H]1O)C)OC(=O)[C@H](COC)N(C)C)[C@@H](C)[C@H]([C@@H](C)CC[C@H]([C@H](C)[C@@H]([C@H](C)/C=C/N(C)C=O)OC(=O)C)OC(=O)[C@H](C)N(C)C)O)OC)C)/C)OC',
#     r'C[C@H]([C@@H](OC)C[C@@H]1OC(=O)C[C@H](C[C@H](/C=C/C=C/C[C@@H](C2=COC(=N2)C[C@@H](C([C@@H](OC(=O)C[C@H](C[C@H](/C=C/C=C/C[C@@H](C3=COC(=N3)C[C@@H](C1(C)C)O)OC)OC)O)C[C@H](OC)[C@H](CCC(=O)[C@@H]([C@H](OC)C/C=C/N(C=O)C)C)C)(C)C)O)OC)OC)O)CCC(=O)[C@@H]([C@H](OC)C/C=C/N(C=O)C)C',
#     r'CC1C(CC=CC2=NC(=CO2)C3=NC(=CO3)C4=NC(=CO4)C(C(C(=O)C=CCC(CC(=O)OC1CC(C(C)CCC(C(C)C(C(C)C=CN(C)C=O)OC(=O)C)OC(=O)C(COC)OC)OC)O)C)OC)OC',
#     r'C[C@@H]1[C@H]([C@@H]([C@H]([C@@H](O1)OC2/C=C\\C=C\\C3=COC(=N3)C(C(CC(/C=C/C=C/C=C\\C(C(OC(=O)/C=C\\C=C\\C=C/C(=C/C2C)/C)C(C)C(CC(C)O)O)C)O)OC)C)OC)OC)O',
#     r'C[C@@H]1[C@H](C[C@@H]2CC=C[C@H](O2)C[C@@H](C/C=C(\\C=C/C(=O)O[C@@H]([C@H]([C@@H](C[C@@H]1OC)OC)C)[C@@H](C)[C@H]([C@@H](C)CCC(=O)[C@H](C)[C@@H]([C@H](C)/C=C/N(C)C=O)OC)O)/C)O)OC',
#     r'C[C@H]1C[C@@H](CC(=O)O[C@H]([C@@H]([C@H](C/C=C/C2=NC(=CO2)C3=NC(=CO3)C4=NC(=CO4)[C@@H]([C@H]([C@@H](C1)O)C)OC)OC)C)C[C@H]([C@@H](C)CCC(=O)[C@H](C)[C@@H]([C@H](C)/C=C/N(C)C=O)OC)OC)OC(=O)N',
#     r'CC1CC/C=C\\C2=NC(=CO2)C3=NC(=CO3)C4=NC(=CO4)C(C(C(=O)/C=C/CC(CC(=O)OC1CC(C(C)CCC(=O)C(C)C(C(C)/C=C/N(C)C=O)OC)OC)O)C)OC',
#     r'C[C@H]1/C=C/[C@H](C[C@@H]([C@H]([C@H]2CC(=CC(=O)O2)C[C@@H](C/C(=C/C=C/C(=O)O[C@@H]([C@H](/C=C/[C@@H](C[C@@H]1OC)OC)C)[C@@H](C)[C@H]([C@@H](C)CCC(=O)[C@H](C)[C@@H](C/C=C/N(C)C=O)OC)OC)/C)OC)C)OC)OC',
#     r'C[C@H]1C/C(=C/[C@H]([C@H](OC(=O)[C@H]([C@@H](NC(=O)[C@H](N(C(=O)[C@@H](NC1=O)C)C)CC2=CNC3=CC=CC=C32)C4=CC=C(C=C4)O)OC)C)C)/C']
#
# ids = ['4752',
#        '9831636',
#        '10438804',
#        '5311281',
#        '5458428',
#        '25077966',
#        '445420',
#        '46937020',
#        '11840920',
#        '25198043',
#        '4267',
#        '21594703',
#        '101168554',
#        '155926072',
#        '6444281',
#        '5289284',
#        '46217451']
# is name is main
if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--split_total", type=int, default=15)
    parser.add_argument("--split_index", type=int, default=0)

    parser.add_argument("--batch_size", type=int, default=1024)

    args = parser.parse_args()
    split_total = args.split_total

    split_index = args.split_index
    batch_size = args.batch_size

    results_file = f"results_{split_index}.csv"

    with open("data/CID-SMILES") as f:
        lines = f.read().splitlines()[1:]

        split_size = len(lines) // split_total
        lines = lines[split_index * split_size: (split_index + 1) * split_size]

        molecules_seqs = [line.split()[1] for line in lines]
        ids = [line.split()[0] for line in lines]

    protein_models = ["ProtBert", "esm3"]
    molecule_models = ["MoLFormer", "ChemBERTa"]
    models_dict = {}
    for protein_model in protein_models:
        models_dict[protein_model] = get_model(protein_model)

    for molecule_model in molecule_models:
        models_dict[molecule_model] = get_model(molecule_model)

    model_pairs = [(protein_model, molecule_model) for protein_model in protein_models for molecule_model in
                   molecule_models]

    pair_to_affinity_model = {}
    for protein_model, molecule_model in model_pairs:
        affinity_model = get_affinity_model(protein_model, molecule_model)
        pair_to_affinity_model[(protein_model, molecule_model)] = affinity_model

    with open(results_file, "w") as f:
        f.write("id,")
        f.write(",".join([f"{protein_model}_{molecule_model}" for protein_model, molecule_model in model_pairs]))
        f.write("\n")

    pbar = tqdm(range(0, len(molecules_seqs), batch_size))
    for batch in pbar:
        batch_molecules = molecules_seqs[batch:batch + batch_size]
        batch_ids = ids[batch:batch + batch_size]
        all_scores = []
        for protein_model, molecule_model in model_pairs:
            affinity_model = pair_to_affinity_model[(protein_model, molecule_model)]
            scores = get_scores(protein_seq, batch_molecules, protein_model, molecule_model, affinity_model)
            all_scores.append(scores)
        with open(results_file, "a") as f:
            for i, id in enumerate(batch_ids):
                f.write(id + ",")
                f.write(",".join([str(score[i]) for score in all_scores]))
                f.write("\n")
        pbar.set_postfix_str(f"Processed {batch + batch_size} molecules", refresh=True)
