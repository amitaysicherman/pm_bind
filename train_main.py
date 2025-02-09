import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from transformers.modeling_outputs import ModelOutput
from transformers import Trainer, TrainingArguments
from seq_to_vec import model_to_dim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BindingDataset(Dataset):
    def __init__(self, protein_model, molecule_model, dataset='data/dataset.csv', data=None):
        self.protein_model = protein_model
        self.molecule_model = molecule_model
        if data is not None:
            self.dataset = data
        else:
            self.dataset = pd.read_csv(dataset)

        self.protein_vecs = np.load(f"data/{protein_model}_vec.npy")
        self.molecule_vecs = np.load(f"data/{molecule_model}_vec.npy")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        protein_idx = int(self.dataset.iloc[idx]['protein_index'])
        ligand_idx = int(self.dataset.iloc[idx]['ligand_index'])
        affinity = float(self.dataset.iloc[idx]['neg_log10_affinity_M'])
        protein_vec = self.protein_vecs[protein_idx]
        molecule_vec = self.molecule_vecs[ligand_idx]
        return {"protein_features": protein_vec, "molecule_features": molecule_vec, "labels": affinity}


def dataset_new_protein_split(bining_dataset: BindingDataset):
    dataset = bining_dataset.dataset
    proteins = dataset['protein_index'].unique()
    np.random.shuffle(proteins)
    train_proteins = proteins[:int(0.9 * len(proteins))]
    test_proteins = proteins[int(0.9 * len(proteins)):]
    train_data = dataset[dataset['protein_index'].isin(train_proteins)]
    test_data= dataset[dataset['protein_index'].isin(test_proteins)]
    train_dataset= BindingDataset(protein_model=bining_dataset.protein_model, molecule_model=bining_dataset.molecule_model,
                            data=train_data)
    test_dataset = BindingDataset(protein_model=bining_dataset.protein_model, molecule_model=bining_dataset.molecule_model,
                            data=test_data)
    return train_dataset, test_dataset

class BindingModel(torch.nn.Module):
    def __init__(self, protein_dim, molecule_dim,dropout=0.2):
        super().__init__()
        # Projection layers
        self.protein_layers = torch.nn.Sequential(
            torch.nn.Dropout(dropout),
            torch.nn.Linear(protein_dim, 64),
            torch.nn.ReLU()
        )
        self.molecule_layers = torch.nn.Sequential(
            torch.nn.Dropout(dropout),
            torch.nn.Linear(molecule_dim, 64),
            torch.nn.ReLU()
        )

        self.final_layers = torch.nn.Sequential(

            # torch.nn.Linear(128, 128),
            torch.nn.BatchNorm1d(128, momentum=0.9, eps=0.001),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(128, 1),
            # torch.nn.ReLU(),
            # torch.nn.Linear(64, 1)
        )

    def forward(self, protein_features, molecule_features, labels=None):
        protein_features = self.protein_layers(protein_features)
        molecule_features = self.molecule_layers(molecule_features)
        combined = torch.cat((protein_features, molecule_features), dim=1)
        out = self.final_layers(combined)
        if labels is None:
            return ModelOutput(logits=out)
        loss = torch.nn.functional.mse_loss(out.squeeze(1), labels)
        return ModelOutput(loss=loss, logits=out)


def main(protein_name, molecule_name, batch_size, lr):
    dataset = BindingDataset(protein_name, molecule_name)

    train_dataset, test_dataset = dataset_new_protein_split(dataset)
    model = BindingModel(model_to_dim[protein_name], model_to_dim[molecule_name]).to(device)
    suffix = f"{protein_name}_{molecule_name}_{batch_size}_{lr}"
    trainer_args = TrainingArguments(
        output_dir=f'./results_{suffix}',
        num_train_epochs=10,
        warmup_steps=100,
        logging_steps=250,
        eval_steps=1_000,
        save_steps=1_000,
        save_total_limit=3,
        save_only_model=True,
        evaluation_strategy="steps",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        lr_scheduler_type='constant',
        logging_dir=f'./logs_{suffix}',
        learning_rate=lr
    )
    trainer = Trainer(
        model=model,
        args=trainer_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset
    )
    trainer.train()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train binding model')
    parser.add_argument('--protein_model', type=str, help='Model to use for protein', default="ProtBert")
    parser.add_argument('--molecule_model', type=str, help='Model to use for molecule', default="MoLFormer")
    parser.add_argument('--batch_size', type=int, help='Batch size', default=1024)
    parser.add_argument('--lr', type=float, help='Learning rate', default=1e-4)
    args = parser.parse_args()
    main(args.protein_model, args.molecule_model, args.batch_size, args.lr)
