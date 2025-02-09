from datasets import load_dataset
from tqdm import tqdm

fasta_all = dict()
smiles_all = dict()
indexes_dataset = []
ds = load_dataset("jglaser/binding_affinity", split='train')
for data in tqdm(ds):
    fasta = data['seq']
    smiles = data['smiles_can']
    if fasta not in fasta_all:
        fasta_all[fasta] = len(fasta_all)
    if smiles not in smiles_all:
        smiles_all[smiles] = len(smiles_all)
    indexes_dataset.append((fasta_all[fasta], smiles_all[smiles], data['neg_log10_affinity_M']))
protein_fasta_file = "data/protein.fasta"
molecule_smiles_file = "data/ligands.smi"
dataset_file = "data/dataset.csv"

print(f"Number of unique proteins: {len(fasta_all)}")
print(f"Number of unique ligands: {len(smiles_all)}")

fasta_by_index = {idx: fasta for fasta, idx in fasta_all.items()}
smiles_by_index = {idx: smiles for smiles, idx in smiles_all.items()}

with open(protein_fasta_file, 'w') as f:
    for inx in range(len(fasta_all)):
        f.write(fasta_by_index[inx] + '\n')
with open(molecule_smiles_file, 'w') as f:
    for inx in range(len(smiles_all)):
        f.write(f"{smiles_by_index[inx]}\n")
with open(dataset_file, 'w') as f:
    f.write("protein_index,ligand_index,neg_log10_affinity_M\n")
    for fasta, smiles, affinity in indexes_dataset:
        f.write(f"{fasta},{smiles},{affinity}\n")
