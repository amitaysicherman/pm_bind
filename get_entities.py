from datasets import load_dataset
from tqdm import tqdm

fasta_all = set()
smiles_all = set()

ds = load_dataset("jglaser/binding_affinity", split='train')
for data in tqdm(ds):

    fasta = data['seq']
    smiles = data['smiles']
    if fasta not in fasta_all:
        fasta_all.add(fasta)
    if smiles not in smiles_all:
        smiles_all.add(smiles)
protein_fasta_file = "data/protein.fasta"
molecule_smiles_file = "data/ligands.smi"

print(f"Number of unique proteins: {len(fasta_all)}")
print(f"Number of unique ligands: {len(smiles_all)}")
with open(protein_fasta_file, 'w') as f:
    for fasta in fasta_all:
        f.write(fasta + '\n')
with open(molecule_smiles_file, 'w') as f:
    for smiles in smiles_all:
        f.write(smiles + '\n')
