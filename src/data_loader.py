"""Load + inspect the raw Twitter CSV without assuming schema."""
import pandas as pd
from pathlib import Path


def load_dataset(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8", engine="python", on_bad_lines="skip")
    print(f"Dataset shape: {df.shape}")
    print("Columns:", df.columns.tolist())
    return df


def show_basic_statistics(df: pd.DataFrame) -> None:
    print("\nMissing values:\n", df.isnull().sum())
    print("\nDuplicate rows:", int(df.duplicated().sum()))
    print("\nDtypes:\n", df.dtypes)
    # Try to guess brand / author / thread columns without crashing
    for col in df.columns:
        try:
            nunique = df[col].nunique(dropna=True)
            print(f"  {col}: nunique={nunique} sample={df[col].dropna().iloc[:2].tolist()}")
        except Exception:
            pass
