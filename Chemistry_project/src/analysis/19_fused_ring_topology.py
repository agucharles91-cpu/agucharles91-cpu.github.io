from pathlib import Path

import pandas as pd
from rdkit import Chem

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "molecular_features.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "fused_ring_features.csv"
REPORT_PATH = PROJECT_ROOT / "reports" / "fused_ring_topology.txt"

# Known example compounds from the structure-pair analysis, used to
# sanity-check the descriptor against chemistry we already understand.
CHECK_COMPOUNDS = {
    "A-5538": "anthracene (expect max_fused=3)",
    "E-1036": "naphthacene (expect max_fused=4)",
    "E-52": "fluorene (expect max_fused=2, one ring is non-aromatic cyclopentane)",
    "B-3585": "benzo(b)fluorene (expect max_fused=3, similar caveat)",
    "C-780": "fluoranthene (expect max_fused>=3, non-6-membered ring present)",
    "B-868": "benzo(k)fluoranthene (expect max_fused higher than C-780)",
    "B-2770": "3,3',5,5'-tetrachlorobiphenyl (expect max_fused=1, two UNFUSED rings)",
    "H-482": "1,3,5-trichlorobenzene (expect max_fused=1)",
    "B-3999": "2-methylnaphthalene (expect max_fused=2)",
    "B-3588": "2-methylanthracene (expect max_fused=3)",
}


def get_fused_aromatic_systems(mol):
    """
    Returns a list of fused aromatic ring-system sizes for a molecule.

    Fusion adjacency is computed across ALL rings (aromatic or not),
    since a non-strictly-aromatic bridging ring (e.g. the 5-membered
    ring in fluoranthene) can still connect two genuinely aromatic
    rings into one rigid, planar, conjugated system.

    A ring counts as "aromatic" (for sizing purposes) only if ALL its
    bonds are aromatic (matching RDKit's own CalcNumAromaticRings
    logic) — this excludes bridging rings like fluoranthene's central
    5-ring from the size count itself, even though that ring still
    participates in connecting the system.

    Returns the count of strictly-aromatic rings within each connected
    fused component that contains at least one aromatic ring.
    """
    ring_info = mol.GetRingInfo()
    atom_rings = ring_info.AtomRings()
    bond_rings = ring_info.BondRings()

    n_rings = len(atom_rings)

    is_aromatic_ring = [
        all(mol.GetBondWithIdx(b).GetIsAromatic() for b in bond_ring)
        for bond_ring in bond_rings
    ]

    if not any(is_aromatic_ring):
        return []

    # Build adjacency across ALL rings (shared >=2 atoms = fused)
    adjacency = {i: set() for i in range(n_rings)}
    for i in range(n_rings):
        for j in range(i + 1, n_rings):
            shared_atoms = set(atom_rings[i]) & set(atom_rings[j])
            if len(shared_atoms) >= 2:
                adjacency[i].add(j)
                adjacency[j].add(i)

    # Find connected components across all rings via BFS
    visited = set()
    systems = []

    for start in range(n_rings):
        if start in visited:
            continue

        component = set()
        queue = [start]
        while queue:
            current = queue.pop()
            if current in component:
                continue
            component.add(current)
            queue.extend(adjacency[current] - component)

        visited |= component

        # Only report components that contain at least one truly
        # aromatic ring; size = count of aromatic rings within it
        aromatic_count_in_component = sum(1 for idx in component if is_aromatic_ring[idx])
        if aromatic_count_in_component > 0:
            systems.append(aromatic_count_in_component)

    return systems

def main():
    print("=" * 70)
    print("FUSED AROMATIC RING TOPOLOGY")
    print("=" * 70)

    df = pd.read_csv(FEATURES_PATH)
    print(f"\nRows: {len(df):,}")

    print("\nComputing fused-ring topology for all compounds...")

    max_fused_sizes = []
    n_systems_list = []

    for smiles in df["SMILES"]:
        mol = Chem.MolFromSmiles(smiles) if pd.notna(smiles) else None

        if mol is None:
            max_fused_sizes.append(0)
            n_systems_list.append(0)
            continue

        systems = get_fused_aromatic_systems(mol)

        if not systems:
            max_fused_sizes.append(0)
            n_systems_list.append(0)
        else:
            max_fused_sizes.append(max(systems))
            n_systems_list.append(len(systems))

    df["max_fused_aromatic_ring_size"] = max_fused_sizes
    df["n_fused_aromatic_systems"] = n_systems_list

    print("\nDistribution of max_fused_aromatic_ring_size:")
    print(df["max_fused_aromatic_ring_size"].value_counts().sort_index().to_string())

    print("\nDistribution of n_fused_aromatic_systems:")
    print(df["n_fused_aromatic_systems"].value_counts().sort_index().to_string())

    # ------------------------------------------------------------
    # SANITY CHECK ON KNOWN COMPOUNDS
    # ------------------------------------------------------------

    print("\n" + "=" * 70)
    print("SANITY CHECK — KNOWN COMPOUNDS")
    print("=" * 70)

    check_rows = []

    for compound_id, description in CHECK_COMPOUNDS.items():
        match = df[df["ID"] == compound_id]
        if match.empty:
            print(f"\n{compound_id}: NOT FOUND in Population C (may have been filtered out)")
            continue

        row = match.iloc[0]
        line = (
            f"\n{compound_id} — {description}\n"
            f"  max_fused_aromatic_ring_size = {row['max_fused_aromatic_ring_size']}\n"
            f"  n_fused_aromatic_systems     = {row['n_fused_aromatic_systems']}\n"
            f"  rdkit_aromatic_rings (old)   = {row['rdkit_aromatic_rings']}\n"
            f"  rdkit_ring_count (old)       = {row['rdkit_ring_count']}"
        )
        print(line)
        check_rows.append(line)

    # ------------------------------------------------------------
    # SAVE
    # ------------------------------------------------------------

    df.to_csv(OUTPUT_PATH, index=False)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("FUSED AROMATIC RING TOPOLOGY\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Rows analyzed: {len(df):,}\n\n")

        f.write("DISTRIBUTION: max_fused_aromatic_ring_size\n" + "-" * 70 + "\n")
        f.write(df["max_fused_aromatic_ring_size"].value_counts().sort_index().to_string())
        f.write("\n\n")

        f.write("DISTRIBUTION: n_fused_aromatic_systems\n" + "-" * 70 + "\n")
        f.write(df["n_fused_aromatic_systems"].value_counts().sort_index().to_string())
        f.write("\n\n")

        f.write("SANITY CHECK — KNOWN COMPOUNDS\n" + "-" * 70 + "\n")
        f.write("\n".join(check_rows))
        f.write("\n")

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print(f"\nData saved to:\n{OUTPUT_PATH}")
    print(f"\nReport saved to:\n{REPORT_PATH}")


if __name__ == "__main__":
    main()