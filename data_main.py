from typing import List, Optional
import typer
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings

# Import functions from other modules
from downloads import download_all_kaggle, download_all_uci, setup_kaggle_credentials



class Settings(BaseSettings):
    """Application settings for data pipeline paths."""
    raw_dir: Path = Field(
        default=Path.cwd() / "raw",
        description="Directory for raw datasets."
    )
    clean_dir: Path = Field(
        default=Path.cwd() / "data" / "clean",
        description="Directory for cleaned datasets."
    )
    clean_master_file: Path = Field(
        default=Path.cwd() / "data" / "clean" / "cleaned_master.csv",
        description="The canonical cleaned dataset file."
    )
    
    class Config:
        env_file = ".env"

settings = Settings()
app = typer.Typer()

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
    # if kaggle:
    #     download_all_kaggle()
    download_all_uci()
    typer.echo(f"Downloads complete. Check {settings.raw_dir.relative_to(Path.cwd())}/")


if __name__ == "__main__":
    app()