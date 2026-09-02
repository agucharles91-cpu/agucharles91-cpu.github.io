import pandas as pd
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRIAGE_PATH = PROJECT_ROOT / "data" / "processed" / "outlier_triage_popc.csv"
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "molecular_features.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "nearest_neighbor_pairs.csv"
REPORT_PATH = PROJECT_ROOT / "reports" / "nearest_neighbor_pairs.txt"

N_NEIGHBORS = 3
SIMILARITY_THRESHOLD = 0.60  # below this, "structurally similar" isn't a defensible claim


def main():
    print("=" * 70)
    print("NEAREST-NEIGHBOR TANIMOTO ANALYSIS — BUCKET 2 OUTLIERS")
    print("=" * 70)

    triage = pd.read_csv(TRIAGE_PATH)
    features = pd.read_csv(FEATURES_PATH)

    print(f"\nTriage rows:   {len(triage):,}")
    print(f"Feature rows:  {len(features):,}")

    # Need SMILES — bring it in from features (triage doesn't carry it)
    smiles_lookup = features[["ID", "SMILES", "Name"]].drop_duplicates("ID")
    triage = triage.merge(smiles_lookup, on="ID", how="left")

    # Full comparison pool: every compound in Population C with valid SMILES
    pool = features[["ID", "SMILES", "Name", "Solubility"]].dropna(subset=["SMILES"]).copy()

    bucket2 = triage[triage["bucket"] == "2_high_reliability_surprising"].copy()
    bucket2 = bucket2.dropna(subset=["SMILES"])

    print(f"\nBucket 2 compounds with valid SMILES: {len(bucket2):,}")
    print(f"Comparison pool size: {len(pool):,}")

    # ------------------------------------------------------------
    # BUILD FINGERPRINTS
    # ------------------------------------------------------------

    print("\nGenerating fingerprints for comparison pool...")

    pool_fps = {}
    for row in pool.itertuples():
        mol = Chem.MolFromSmiles(row.SMILES)
        if mol is not None:
            pool_fps[row.ID] = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)

    print(f"Valid pool fingerprints: {len(pool_fps):,}")

    # ------------------------------------------------------------
    # FOR EACH BUCKET-2 COMPOUND, FIND NEAREST NEIGHBORS
    # ------------------------------------------------------------

    print("\nSearching for nearest structural neighbors...")

    results = []

    pool_solubility = pool.set_index("ID")["Solubility"].to_dict()
    pool_name = pool.set_index("ID")["Name"].to_dict()

    for row in bucket2.itertuples():
        outlier_id = row.ID
        mol = Chem.MolFromSmiles(row.SMILES)
        if mol is None:
            continue

        outlier_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)

        sims = []
        for other_id, other_fp in pool_fps.items():
            if other_id == outlier_id:
                continue
            sim = DataStructs.TanimotoSimilarity(outlier_fp, other_fp)
            sims.append((other_id, sim))

        sims.sort(key=lambda x: -x[1])
        top_neighbors = sims[:N_NEIGHBORS]

        for rank, (neighbor_id, sim) in enumerate(top_neighbors, start=1):
            neighbor_solubility = pool_solubility.get(neighbor_id)
            solubility_gap = abs(row.Solubility - neighbor_solubility) if neighbor_solubility is not None else None

            results.append({
                "outlier_id": outlier_id,
                "outlier_name": row.Name,
                "outlier_solubility": row.Solubility,
                "outlier_residual": row.Residual,
                "neighbor_rank": rank,
                "neighbor_id": neighbor_id,
                "neighbor_name": pool_name.get(neighbor_id),
                "neighbor_solubility": neighbor_solubility,
                "tanimoto_similarity": sim,
                "solubility_gap": solubility_gap,
            })

    results_df = pd.DataFrame(results)

    print(f"\nNeighbor pairs generated: {len(results_df):,}")

    # ------------------------------------------------------------
    # SORT AND SUMMARIZE — most compelling pairs first
    # ------------------------------------------------------------
    # Compelling = high similarity AND large solubility gap.
    # Pairs below the similarity threshold are excluded from the
    # "structurally similar" narrative — they are simply the closest
    # available match in the pool, not a genuinely similar structure.

    rank1 = results_df[results_df["neighbor_rank"] == 1]

    n_outliers_total = bucket2["ID"].nunique()
    n_outliers_with_close_match = rank1[rank1["tanimoto_similarity"] >= SIMILARITY_THRESHOLD]["outlier_id"].nunique()

    print(
        f"\nOutliers with a close structural match "
        f"(rank-1 similarity >= {SIMILARITY_THRESHOLD}): "
        f"{n_outliers_with_close_match} of {n_outliers_total}"
    )

    top_1_per_outlier = (
        rank1[rank1["tanimoto_similarity"] >= SIMILARITY_THRESHOLD]
        .sort_values("solubility_gap", ascending=False)
    )

    below_threshold = (
        rank1[rank1["tanimoto_similarity"] < SIMILARITY_THRESHOLD]
        .sort_values("tanimoto_similarity", ascending=False)
    )

    print("\n" + "=" * 70)
    print(f"STRIKING NEIGHBOR PAIRS — similarity >= {SIMILARITY_THRESHOLD} (by solubility gap)")
    print("=" * 70)
    print(
        top_1_per_outlier[[
            "outlier_id", "outlier_solubility", "neighbor_id",
            "neighbor_solubility", "tanimoto_similarity", "solubility_gap",
        ]]
        .to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )

    print("\n" + "=" * 70)
    print(f"OUTLIERS WITH NO CLOSE MATCH — best available neighbor below {SIMILARITY_THRESHOLD}")
    print("=" * 70)
    print(
        below_threshold[[
            "outlier_id", "outlier_solubility", "neighbor_id",
            "neighbor_solubility", "tanimoto_similarity", "solubility_gap",
        ]]
        .head(15)
        .to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )

    # ------------------------------------------------------------
    # SAVE
    # ------------------------------------------------------------

    results_df.to_csv(OUTPUT_PATH, index=False)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("NEAREST-NEIGHBOR TANIMOTO ANALYSIS — BUCKET 2 OUTLIERS\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Bucket 2 compounds analyzed: {len(bucket2):,}\n")
        f.write(f"Comparison pool size: {len(pool_fps):,}\n")
        f.write(f"Neighbors per outlier: {N_NEIGHBORS}\n")
        f.write(f"Similarity threshold for 'structurally similar' claim: {SIMILARITY_THRESHOLD}\n")
        f.write(
            f"Outliers with a close structural match: "
            f"{n_outliers_with_close_match} of {n_outliers_total}\n\n"
        )

        f.write(f"STRIKING NEIGHBOR PAIRS — similarity >= {SIMILARITY_THRESHOLD}\n")
        f.write("-" * 70 + "\n")
        f.write(
            top_1_per_outlier[[
                "outlier_id", "outlier_solubility", "neighbor_id",
                "neighbor_solubility", "tanimoto_similarity", "solubility_gap",
            ]]
            .to_string(index=False, float_format=lambda x: f"{x:.4f}")
        )
        f.write("\n\n")

        f.write(f"OUTLIERS WITH NO CLOSE MATCH — best neighbor below {SIMILARITY_THRESHOLD}\n")
        f.write("-" * 70 + "\n")
        f.write(
            below_threshold[[
                "outlier_id", "outlier_solubility", "neighbor_id",
                "neighbor_solubility", "tanimoto_similarity", "solubility_gap",
            ]]
            .to_string(index=False, float_format=lambda x: f"{x:.4f}")
        )
        f.write("\n\n")

        f.write("FULL NEIGHBOR TABLE (all ranks, all similarities)\n")
        f.write("-" * 70 + "\n")
        f.write(
            results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}")
        )
        f.write("\n")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nData saved to:\n{OUTPUT_PATH}")
    print(f"\nReport saved to:\n{REPORT_PATH}")


if __name__ == "__main__":
    main()