from __future__ import annotations
import shutil
import subprocess
from typing import List, Dict
import requests
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings
import os


class Settings(BaseSettings):
    """Application settings for data download paths and configurations."""
    raw_dir: Path = Field(
        default=Path.cwd() / "raw",
        description="Directory for raw datasets."
    )
    kaggle_datasets: Dict[str, str] = {
        "disease_symptoms": "choongqianzheng/disease-and-symptoms-dataset",
        "symptom2disease": "niyarrbarman/symptom2disease",
        "descriptions_weights": "itachi9604/disease-symptom-description-dataset",
    }
    uci_links: Dict[str, str] = {
    "heart_disease": "https://archive.ics.uci.edu/dataset/45/heart+disease",
    "diabetes": "https://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+Hospitals+for+Years+1999-2008",
    "liver_disease": "https://archive.ics.uci.edu/ml/machine-learning-databases/00225/indian_liver_patient.csv",
    "kidney_disease": "https://archive.ics.uci.edu/ml/machine-learning-databases/00383/kidney_disease.csv",
    "breast_cancer": "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wpbc.data",
    "parkinsons": "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data",
    "thyroid_disease": "https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/ann-train.data",
    "hepatitis": "https://archive.ics.uci.edu/ml/machine-learning-databases/hepatitis/hepatitis.data",
    "tuberculosis": "https://archive.ics.uci.edu/ml/machine-learning-databases/00462/tuberculosis.csv",
    "covid19": "https://archive.ics.uci.edu/ml/machine-learning-databases/00529/COVID-19%20Dataset.zip",
    "pneumonia": "https://archive.ics.uci.edu/ml/machine-learning-databases/00454/pneumonia.csv",
    "malaria": "https://archive.ics.uci.edu/ml/machine-learning-databases/00465/malaria.csv",
    "heart_failure": "https://archive.ics.uci.edu/ml/machine-learning-databases/00519/heart_failure_clinical_records_dataset.csv",
    "stroke": "https://archive.ics.uci.edu/ml/machine-learning-databases/00514/healthcare-dataset-stroke-data.csv",
    "sepsis": "https://archive.ics.uci.edu/ml/machine-learning-databases/00496/sepsis_data.csv",
    "chronic_kidney_disease": "https://archive.ics.uci.edu/ml/datasets/Chronic+Kidney+Disease",
    "lupus": "https://archive.ics.uci.edu/ml/machine-learning-databases/00451/lupus.csv",
    "alzheimers": "https://archive.ics.uci.edu/ml/machine-learning-databases/00462/alzheimer.csv",
    "asthma": "https://archive.ics.uci.edu/ml/machine-learning-databases/00472/asthma.csv",
    "autism": "https://archive.ics.uci.edu/ml/machine-learning-databases/00433/autism.csv",
    "epilepsy": "https://archive.ics.uci.edu/ml/machine-learning-databases/00423/epilepsy.csv",
    "fibromyalgia": "https://archive.ics.uci.edu/ml/machine-learning-databases/00455/fibromyalgia.csv",
    "gastroenteritis": "https://archive.ics.uci.edu/ml/machine-learning-databases/00463/gastroenteritis.csv",
    "hearing_loss": "https://archive.ics.uci.edu/ml/machine-learning-databases/00473/hearing_loss.csv",
    "migraine": "https://archive.ics.uci.edu/ml/machine-learning-databases/00474/migraine.csv",
    "osteoporosis": "https://archive.ics.uci.edu/ml/machine-learning-databases/00475/osteoporosis.csv",
    "psoriasis": "https://archive.ics.uci.edu/ml/machine-learning-databases/00476/psoriasis.csv",
    "rheumatoid_arthritis": "https://archive.ics.uci.edu/ml/machine-learning-databases/00477/rheumatoid_arthritis.csv",
    "sickle_cell_disease": "https://archive.ics.uci.edu/ml/machine-learning-databases/00478/sickle_cell_disease.csv",
    "systemic_lupus_erythematosus": "https://archive.ics.uci.edu/ml/machine-learning-databases/00479/systemic_lupus_erythematosus.csv",
    "ulcerative_colitis": "https://archive.ics.uci.edu/ml/machine-learning-databases/00480/ulcerative_colitis.csv",
    "vitiligo": "https://archive.ics.uci.edu/ml/machine-learning-databases/00481/vitiligo.csv",
    "uci_symptoms_1": "https://archive.ics.uci.edu/ml/machine-learning-databases/00482/symptoms.csv",
    "uci_symptoms_2": "https://archive.ics.uci.edu/ml/machine-learning-databases/00359/symptoms.csv",
    "uci_diseases_1": "https://archive.ics.uci.edu/ml/machine-learning-databases/00359/diseases.csv",
    "uci_diseases_2": "https://archive.ics.uci.edu/ml/machine-learning-databases/00482/diseases.csv",
    "uci_treatments_1": "https://archive.ics.uci.edu/ml/machine-learning-databases/00359/treatments.csv",
    "uci_treatments_2": "https://archive.ics.uci.edu/ml/machine-learning-databases/00482/treatments.csv",
}

    class Config:
        env_file = ".env"

settings = Settings()

def setup_kaggle_credentials(source_path: Path):
    """
    Copies the kaggle.json file to the standard location.
    
    Args:
        source_path: The path to the kaggle.json file.
    """
    # Define the destination directory for the credentials file
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(parents=True, exist_ok=True)
    
    # Define the destination path for the kaggle.json file
    dest_path = kaggle_dir / "kaggle.json"
    
    # Copy the file to the destination
    shutil.copy(source_path, dest_path)
    print(f"Kaggle credentials copied to: {dest_path}")


def download_kaggle(slug: str, dest_dir: Path = settings.raw_dir) -> List[Path]:
    """Download using kaggle CLI. Requires KAGGLE_USERNAME & KAGGLE_KEY env vars or ~/.kaggle.json."""
    target = dest_dir / slug.replace("/", "_")
    target.mkdir(parents=True, exist_ok=True)

    # 2. Define the path for the temporary zip file
    zip_filename = f"{slug.split('/')[-1]}.zip"
    zip_path = target / zip_filename

    try:
        subprocess.check_call(
            ["kaggle", "datasets", "download", "-d", slug, "-p", str(target), "--unzip"]
            )
        print("Download successful! 🎉")
        return list(target.glob('*'))
    except Exception as e:
        print("kaggle download failed:", e)

    if zip_path.exists():
        os.remove(zip_path)
        print("Cleaned up temporary zip file.")
        return []

def download_uci(name: str, url: str, dest_dir: Path = settings.raw_dir) -> Path:
    """Download a file from a UCI URL."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    local = dest_dir / f"{name}.csv"
    if local.exists():
        return local
    print(f"Downloading UCI file {name} from {url}")
    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        local.write_bytes(r.content)
        return local
    except requests.exceptions.RequestException as e:
        print(f"Failed to download from {url}: {e}")
        return Path() # Return an empty Path object or handle the error as needed

def download_all_kaggle(dest_dir: Path = settings.raw_dir) -> Dict[str, List[Path]]:
    """Download all datasets from Kaggle."""
    results = {}
    for k, slug in settings.kaggle_datasets.items():
        results[k] = download_kaggle(slug, dest_dir)
    return results

def download_all_uci(dest_dir: Path = settings.raw_dir) -> Dict[str, Path]:
    """Download all datasets from UCI."""
    results = {}
    for k, url in settings.uci_links.items():
        results[k] = download_uci(k, url, dest_dir)
    return results