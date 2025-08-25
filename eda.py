import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from data import guess_target_column


def analisi_esplorativa(df: pd.DataFrame, grafici: bool = False) -> None:
    print("\n=== Statistiche descrittive ===")
    print(df.describe(include="all").transpose())
    print("\n=== Valori Nulli per colonna ===")
    print(df.isna().sum())

    if not grafici:
        return

    try:
        target = guess_target_column(df)
        sns.countplot(x=target, data=df)
        plt.title("Distribuzione classe target")
        plt.show()
    except Exception:
        pass

    corr = df.select_dtypes(include="number").corr(numeric_only=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title("Matrice di correlazione")
    plt.show()
    
    sns.pairplot(df[["Jitter(%)", "Shimmer(dB)", "HNR", target]],
                    hue=target, diag_kind="kde")
    plt.show()
