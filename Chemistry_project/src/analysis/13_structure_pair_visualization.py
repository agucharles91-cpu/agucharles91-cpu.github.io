import pandas as pd
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Draw, AllChem

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PAIRS_PATH = PROJECT_ROOT / "data" / "processed" / "nearest_neighbor_pairs.csv"
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "molecular_features.csv"
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures" / "structure_pairs"
REPORT_PATH = PROJECT_ROOT / "reports" / "structure_pair_inspection.txt"

SIMILARITY_THRESHOLD = 0.60
TOP_N_PAIRS = 10  # how many pairs to render as images


def main():
    print("=" * 70)
    print("STRUCTURE PAIR VISUALIZATION")
    print("=" * 70)

    pairs = pd.read_csv(PAIRS_PATH)
    features = pd.read_csv(FEATURES_PATH)

    # Only rank-1 neighbors above the similarity threshold
    striking = pairs[
        (pairs["neighbor_rank"] == 1)
        & (pairs["tanimoto_similarity"] >= SIMILARITY_THRESHOLD)
    ].copy()

    # Deduplicate symmetric pairs (A->B and B->A both appearing)
    striking["pair_key"] = striking.apply(
        lambda r: tuple(sorted([r["outlier_id"], r["neighbor_id"]])), axis=1
    )
    striking = striking.drop_duplicates("pair_key")

    striking = striking.sort_values("solubility_gap", ascending=False).head(TOP_N_PAIRS)

    print(f"\nRendering top {len(striking)} structure pairs...")

    lookup = features.set_index("ID")[[
        "SMILES", "Name", "rdkit_molwt", "rdkit_mollogp", "rdkit_tpsa",
        "rdkit_hbd", "rdkit_hba", "rdkit_ring_count",
    ]]
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    report_lines = []
    report_lines.append("STRUCTURE PAIR INSPECTION\n" + "=" * 70 + "\n")

    for row in striking.itertuples():
        id_a = row.outlier_id
        id_b = row.neighbor_id

        if id_a not in lookup.index or id_b not in lookup.index:
            print(f"Skipping {id_a}/{id_b} — missing from feature lookup")
            continue

        info_a = lookup.loc[id_a]
        info_b = lookup.loc[id_b]

        mol_a = Chem.MolFromSmiles(info_a["SMILES"])
        mol_b = Chem.MolFromSmiles(info_b["SMILES"])

        if mol_a is None or mol_b is None:
            print(f"Skipping {id_a}/{id_b} — invalid SMILES")
            continue

        flag = " *** IDENTICAL FINGERPRINT ***" if row.tanimoto_similarity >= 0.999 else ""

        header = (
            f"\n{id_a} vs {id_b}  "
            f"(Tanimoto={row.tanimoto_similarity:.4f}, "
            f"solubility gap={row.solubility_gap:.4f}){flag}"
        )
        print(header)
        report_lines.append(header)

        for label, cid, info, sol in [
            ("A", id_a, info_a, row.outlier_solubility),
            ("B", id_b, info_b, row.neighbor_solubility),
        ]:
            line = (
                f"  [{label}] {cid} ({info['Name']}): "
                f"Solubility={sol:.4f}, MolWt={info['rdkit_molwt']:.2f}, "
                f"MolLogP={info['rdkit_mollogp']:.2f}, TPSA={info['rdkit_tpsa']:.2f}, "
                f"HBD={info['rdkit_hbd']}, HBA={info['rdkit_hba']}, "
                f"Rings={info['rdkit_ring_count']}\n  SMILES: {info['SMILES']}"
            )
            print(line)
            report_lines.append(line)

        # Render side-by-side image
        img = Draw.MolsToGridImage(
            [mol_a, mol_b],
            molsPerRow=2,
            subImgSize=(400, 400),
            legends=[
                f"{id_a}  logS={row.outlier_solubility:.2f}",
                f"{id_b}  logS={row.neighbor_solubility:.2f}",
            ],
        )

        img_path = FIGURE_DIR / f"pair_{id_a}_{id_b}.png"
        img.save(img_path)
        print(f"  Saved image: {img_path.name}")
        report_lines.append(f"  Image: {img_path.name}\n")

    REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print(f"\nImages saved to:\n{FIGURE_DIR}")
    print(f"\nReport saved to:\n{REPORT_PATH}")


if __name__ == "__main__":
    main()