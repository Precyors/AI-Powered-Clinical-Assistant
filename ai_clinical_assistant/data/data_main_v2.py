import typer
from cleaning_v2 import process_all_datasets
from embeddings import prepare_for_embeddings
from config_v2 import CLEAN_DIR, RAW_DIR
from pathlib import Path

from downloads import download_all_kaggle, download_all_uci, setup_kaggle_credentials


app = typer.Typer(help="📦 Data Cleaning + Embedding Prep CLI")


@app.command(name="setup")
def setup_credentials(
    file_path: Path = typer.Argument(
        ..., help="Path to your kaggle.json file."
    )
):
    """Sets up Kaggle credentials by copying the JSON file to the correct location."""
    try:
        setup_kaggle_credentials(file_path)
        typer.echo("Kaggle credentials configured successfully!,\n \
                   you can now run `python data_main.py download --kaggle` to download datasets.")
    except FileNotFoundError:
        typer.echo(f"Error: The file at {file_path} was not found.")
        raise typer.Exit(code=1)
    except Exception as e:
        typer.echo(f"An unexpected error occurred: {e}")
        raise typer.Exit(code=1)


# Define the CLI commands
@app.command()
def download(kaggle: bool = False):
    """Download datasets. Requires kaggle CLI & credentials for Kaggle downloads."""
    if kaggle:
        download_all_kaggle()
    download_all_uci()
    typer.echo(f"Downloads complete. Check {RAW_DIR.relative_to(Path.cwd())}/")



@app.command("clean")
def clean_data():
    """Clean raw datasets and save normalized outputs."""
    cleaned_files = process_all_datasets()
    if cleaned_files:
        typer.echo(f"✅ Cleaned {len(cleaned_files)} datasets → {CLEAN_DIR}")

@app.command("prepare-embeddings")
def prepare_embeddings():
    """Prepare cleaned data for embeddings."""
    path = prepare_for_embeddings()
    if path:
        typer.echo(f"✅ Embedding corpus ready → {path}")

@app.command("process-all")
def process_all():
    """Run full pipeline: cleaning + embedding prep."""
    cleaned_files = process_all_datasets()
    if cleaned_files:
        path = prepare_for_embeddings()
        typer.echo(f"🎯 Full pipeline complete → {path}")

if __name__ == "__main__":
    app()
