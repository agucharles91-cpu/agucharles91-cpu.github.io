from pathlib import Path

import pandas as pd
from rdkit import Chem

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "molecular_features.csv"

TARGET_COMPOUNDS = ["E-52", "C-780", "B-3585", "B-868"]


def inspect_rings(compound_id, smiles, name):
    print("\n" + "=" * 70)
    print(f"{compound_id} — {name}")
    print("=" * 70)
    print(f"SMILES: {smiles}")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print("INVALID SMILES — could not parse.")
        return

    ring_info = mol.GetRingInfo()
    atom_rings = ring_info.AtomRings()

    print(f"\nTotal atoms: {mol.GetNumAtoms()}")
    print(f"Total rings found by SSSR: {len(atom_rings)}")

    print("\nPer-atom aromaticity flags:")
    atom_flags = []
    for atom in mol.GetAtoms():
        atom_flags.append(f"  idx={atom.GetIdx():2d}  {atom.GetSymbol():2s}  aromatic={atom.GetIsAromatic()}")
    print("\n".join(atom_flags))

    print("\nPer-ring details:")
    for i, ring in enumerate(atom_rings):
        ring_atoms = [mol.GetAtomWithIdx(a) for a in ring]
        symbols = [a.GetSymbol() for a in ring_atoms]
        aromatic_flags = [a.GetIsAromatic() for a in ring_atoms]
        all_aromatic = all(aromatic_flags)

        print(
            f"  Ring {i}: size={len(ring)}, atoms={list(ring)}, "
            f"symbols={symbols}"
        )
        print(
            f"           per-atom aromatic={aromatic_flags}, "
            f"ALL_AROMATIC={all_aromatic}"
        )

    # Also report RDKit's own aromatic ring count for comparison
    from rdkit.Chem import rdMolDescriptors
    official_aromatic_count = rdMolDescriptors.CalcNumAromaticRings(mol)
    official_ring_count = rdMolDescriptors.CalcNumRings(mol)

    print(f"\nRDKit CalcNumAromaticRings: {official_aromatic_count}")
    print(f"RDKit CalcNumRings:         {official_ring_count}")
    print(f"Our own 'all atoms aromatic' ring count: "
          f"{sum(1 for ring in atom_rings if all(mol.GetAtomWithIdx(a).GetIsAromatic() for a in ring))}")


def main():
    print("=" * 70)
    print("RING PERCEPTION DIAGNOSTIC")
    print("=" * 70)

    df = pd.read_csv(FEATURES_PATH)

    for compound_id in TARGET_COMPOUNDS:
        match = df[df["ID"] == compound_id]
        if match.empty:
            print(f"\n{compound_id}: NOT FOUND in Population C")
            continue

        row = match.iloc[0]
        inspect_rings(compound_id, row["SMILES"], row.get("Name", "unknown"))

    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()