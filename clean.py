import polars as pl
from pathlib import Path
import json

from pathlib import Path
import shutil

from pathlib import Path
import shutil

def migrate_data(old_path: str, new_path: str):
    """
    Migrates data from an old CSV file to a new CSV file by copying it.

    Args:
        old_path (str): Path to the source CSV file.
        new_path (str): Path to the destination CSV file.
    """
    old_file = Path(old_path)
    new_file = Path(new_path)

    # Check if the source file exists
    if not old_file.exists():
        raise FileNotFoundError(f"Error: The file '{old_file}' was not found.")

    # Create the destination folder if it doesn't exist
    new_file.parent.mkdir(parents=True, exist_ok=True)

    # Copy the file to the new location
    shutil.copy(old_file, new_file)

    print(f"Data copied from '{old_file}' to '{new_file}' successfully.")


def clean_medical_data(file_path: str, output_path: str) -> pl.DataFrame:
    """
    Cleans a medical dataset (precaution or symptom data) based on file structure.
    """
    input_file = Path(file_path)
    output_file = Path(output_path)

    csv_files = list(input_file.glob("*.csv"))

    if not csv_files.exists():
        raise FileNotFoundError(f"Error: The file '{input_file}' was not found.")
        return []
    
    for file in csv_files:
        print(f"🧹 Cleaning → {file.name}")
        # Read CSV
        df = pl.read_csv(file, infer_schema_length=100)

        # Detect file type based on column patterns
        precaution_cols = [col for col in df.columns if col.lower().startswith("precaution_")]
        symptom_cols = [col for col in df.columns if col.lower().startswith("symptom_")]
        description_cols = [col for col in df.columns if col.lower().startswith("description")]

        # Normalize disease column
        df = df.with_columns(pl.col("Disease").str.to_lowercase().alias("disease")).drop("Disease")

        if precaution_cols:
            # Process precaution file
            df = df.with_columns(
                pl.concat_str(
                    [pl.col(col).fill_null("") for col in precaution_cols],
                    separator=", "
                ).str.to_lowercase().alias("precautions")
            )
            df = df.drop(precaution_cols)
            df_cleaned = df.fill_null("N/A").unique().sort("disease")

        elif symptom_cols:
            # Process symptom file
            df = df.with_columns(
                pl.concat_str(
                    [pl.col(col).fill_null("") for col in symptom_cols],
                    separator=", "
                ).str.to_lowercase().alias("symptoms")
            )
            df = df.drop(symptom_cols)

            df = df.with_columns(
                pl.col("symptoms")
                .str.replace("high_fever", "hyperthermia")
                .str.replace_all(r"\s*,\s*", ", ")
                .str.strip_chars(", ")
                .alias("symptoms")
            )

            df_cleaned = (
                df
                .with_columns(pl.col("symptoms").str.split(", "))
                .explode("symptoms")
                .group_by("disease")
                .agg(pl.col("symptoms").unique().sort())
                .with_columns(pl.col("symptoms").list.join(", "))
                .sort("disease")
            )
            df_cleaned = df_cleaned.fill_null("N/A")

        elif description_cols:
            df = df.with_columns(
                pl.concat_str(
                    [pl.col(col).fill_null("") for col in description_cols],
                    separator=" "
                ).str.to_lowercase().alias("description")
            )
            df = df.drop(description_cols)
            df = df.unique()
            df_cleaned = df.fill_null("N/A").sort("disease")


        else:
            raise ValueError(f"Unsupported file format: {file_path}")

        # Save cleaned data
        df_cleaned.write_csv(output_file)
        return df_cleaned













def merge_datasets(df_precautions: pl.DataFrame, df_descriptions: pl.DataFrame, df_symptoms: pl.DataFrame, output_path: str) -> pl.DataFrame:
    """
    Merges the precautions dataset with the symptoms dataset on the 'disease' column.
    """
        # 1. Left join df2 onto df1
    intermediate_df = df_symptoms.join(df_precautions, on="disease", how="left")

    # 2. Left join df3 onto the result of the first join
    merged_df = intermediate_df.join(df_descriptions, on="disease", how="left")
    return merged_df

def embedding_ready(df: pl.DataFrame) -> pl.DataFrame:
    """
    Prepares the merged DataFrame for embedding by combining text fields.
    """
    df = df.with_columns([
        pl.col('disease').fill_null("N/A"),
        pl.col('symptoms').fill_null("N/A"),
        pl.col('precautions').fill_null("N/A"),
        pl.col('description').fill_null("N/A"),
        ])

    df = df.with_columns(
    pl.format(
        "Disease: {}. Symptoms: {}. Precautions: {}. Description: {}.",
        pl.col('disease'),
        pl.col('symptoms'),
        pl.col('precautions'),
        pl.col('description')
    ).alias('content')
    )

    json_chunks = df.to_dicts()
    return json_chunks



# migrate_data("./raw/niyarrbarman_symptom2disease/Disease precaution.csv","./raw/Disease precaution.csv")
# migrate_data("./raw/niyarrbarman_symptom2disease/DiseaseAndSymptoms.csv","./raw/DiseaseAndSymptoms.csv")
# migrate_data("./raw/itachi9604_disease-symptom-description-dataset/symptom_Description.csv","./raw/symptom_Description.csv")







# # Paths
# precaution_path = "raw/Disease precaution.csv"
# symptom_path = "raw/DiseaseAndSymptoms.csv"
# cleaned_precautions_path = "cleaned_precautions.csv"
# cleaned_symptoms_path = "cleaned_symptoms.csv"
# merged_path = "merged_disease_data.csv"
# description_path = "raw/symptom_Description.csv"
# cleaned_description_path = "cleaned_descriptions.csv"


if __name__ == "__main__":
    # Clean both datasets
    df_precautions = clean_medical_data(precaution_path, cleaned_precautions_path)
    df_symptoms = clean_medical_data(symptom_path, cleaned_symptoms_path)
    df_descriptions = clean_medical_data(description_path, cleaned_description_path)

    # Merge them
    merged_df = merge_datasets(df_precautions, df_descriptions, df_symptoms, merged_path)
    merged_df.write_csv(merged_path)
    # Prepare for embedding
    json_ready = embedding_ready(merged_df)
    with open("./embeddings_ready/embedding_ready.json", "w") as f:
        json.dump(json_ready, f, indent=2)



    print("\n📌 Final Merged DataFrame:")
    print(merged_df)
