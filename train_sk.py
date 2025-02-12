import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
from typing import Dict, Tuple
import joblib
import os


class BindingDataset:
    def __init__(self, protein_model: str, molecule_model: str, dataset='data/dataset.csv', data=None):
        self.protein_model = protein_model
        self.molecule_model = molecule_model
        if data is not None:
            self.dataset = data
        else:
            self.dataset = pd.read_csv(dataset)

        # Load feature vectors
        self.protein_vecs = np.load(f"data/{protein_model}_vec.npy")
        self.molecule_vecs = np.load(f"data/{molecule_model}_vec.npy")

        # Prepare feature matrix
        self.X, self.y = self._prepare_features()

    def _prepare_features(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare feature matrix and target values."""
        X_list = []
        y_list = []

        for idx in range(len(self.dataset)):
            protein_idx = int(self.dataset.iloc[idx]['protein_index'])
            ligand_idx = int(self.dataset.iloc[idx]['ligand_index'])
            affinity = float(self.dataset.iloc[idx]['neg_log10_affinity_M'])

            # Concatenate protein and molecule features
            features = np.concatenate([
                self.protein_vecs[protein_idx],
                self.molecule_vecs[ligand_idx]
            ])

            X_list.append(features)
            y_list.append(affinity)

        return np.array(X_list), np.array(y_list)


def dataset_new_protein_split(binding_dataset: BindingDataset) -> Tuple[BindingDataset, BindingDataset]:
    """Split dataset by proteins to avoid data leakage."""
    dataset = binding_dataset.dataset
    proteins = dataset['protein_index'].unique()
    np.random.shuffle(proteins)

    train_proteins = proteins[:int(0.9 * len(proteins))]
    test_proteins = proteins[int(0.9 * len(proteins)):]

    train_data = dataset[dataset['protein_index'].isin(train_proteins)]
    test_data = dataset[dataset['protein_index'].isin(test_proteins)]

    train_dataset = BindingDataset(
        protein_model=binding_dataset.protein_model,
        molecule_model=binding_dataset.molecule_model,
        data=train_data
    )

    test_dataset = BindingDataset(
        protein_model=binding_dataset.protein_model,
        molecule_model=binding_dataset.molecule_model,
        data=test_data
    )

    return train_dataset, test_dataset


class MLModelsEnsemble:
    def __init__(self):
        self.models = {
            'rf': RandomForestRegressor(
                n_estimators=100,
                max_depth=2,
                random_state=42
            ),
            'knn': KNeighborsRegressor(
                n_neighbors=5,
                weights='distance'
            ),
            'linear': LinearRegression(),
            'ridge': Ridge(alpha=1.0),
            'lasso': Lasso(alpha=1.0),
            'elastic': ElasticNet(alpha=1.0, l1_ratio=0.5),
            'xgb': xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=2,
                random_state=42
            )
        }
        self.scalers: Dict[str, StandardScaler] = {}
        self.trained_models: Dict[str, bool] = {name: False for name in self.models.keys()}

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train all models with scaled data."""
        for name, model in self.models.items():
            # Create and fit scaler
            self.scalers[name] = StandardScaler()
            X_scaled = self.scalers[name].fit_transform(X_train)

            print(f"Training {name}...")
            model.fit(X_scaled, y_train)
            self.trained_models[name] = True

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Evaluate all trained models."""
        results = {}

        for name, model in self.models.items():
            if not self.trained_models[name]:
                continue

            X_scaled = self.scalers[name].transform(X_test)
            y_pred = model.predict(X_scaled)

            results[name] = {
                'mse': mean_squared_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'r2': r2_score(y_test, y_pred)
            }

        return results

    def save_models(self, output_dir: str):
        """Save trained models and scalers."""
        os.makedirs(output_dir, exist_ok=True)

        for name, model in self.models.items():
            if not self.trained_models[name]:
                continue

            # Save model
            model_path = os.path.join(output_dir, f"{name}_model.joblib")
            joblib.dump(model, model_path)

            # Save scaler
            scaler_path = os.path.join(output_dir, f"{name}_scaler.joblib")
            joblib.dump(self.scalers[name], scaler_path)


def main(protein_name: str, molecule_name: str):
    # Load and split dataset
    dataset = BindingDataset(protein_name, molecule_name)
    train_dataset, test_dataset = dataset_new_protein_split(dataset)

    # Initialize and train models
    ensemble = MLModelsEnsemble()
    ensemble.train(train_dataset.X, train_dataset.y)

    # Evaluate models
    results = ensemble.evaluate(test_dataset.X, test_dataset.y)

    # Print results
    print("\nModel Evaluation Results:")
    for model_name, metrics in results.items():
        print(f"\n{model_name.upper()}:")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")

    # Save models
    output_dir = f'results_{protein_name}_{molecule_name}'
    ensemble.save_models(output_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train binding models using classical ML approaches')
    parser.add_argument('--protein_model', type=str, help='Model to use for protein', default="ProtBert")
    parser.add_argument('--molecule_model', type=str, help='Model to use for molecule', default="MoLFormer")

    args = parser.parse_args()
    main(args.protein_model, args.molecule_model)