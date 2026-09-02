import os
from pathlib import Path

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values


PROJECT_ROOT = Path(__file__).parent.parent.parent
CSV_PATH = PROJECT_ROOT / "data" / "raw" / "curated-solubility-dataset.csv"


def load_data():

    print("=" * 70)
    print("LOADING AQSOLDB INTO POSTGRESQL")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. Load CSV
    # ---------------------------------------------------------

    print("\nReading CSV...")

    df = pd.read_csv(CSV_PATH)

    print(f"Rows found: {len(df):,}")
    print(f"Columns found: {len(df.columns)}")

    # ---------------------------------------------------------
    # 2. Validate expected row count
    # ---------------------------------------------------------

    if len(df) != 9982:
        raise ValueError(
            f"Unexpected row count: {len(df):,}. "
            "Expected 9,982 rows."
        )

    # ---------------------------------------------------------
    # 3. Get database password
    # ---------------------------------------------------------

    password = os.getenv("POSTGRES_PASSWORD")

    if not password:
        raise RuntimeError(
            "POSTGRES_PASSWORD environment variable is not set."
        )

    # ---------------------------------------------------------
    # 4. Connect to PostgreSQL
    # ---------------------------------------------------------

    print("\nConnecting to PostgreSQL...")

    conn = psycopg2.connect(
        host="localhost",
        port=5432,
        database="postgres",
        user="postgres",
        password=password
    )

    print("Connected successfully.")

    try:

        with conn.cursor() as cur:

            # -------------------------------------------------
            # 5. Clear existing data
            # -------------------------------------------------

            print("\nClearing existing tables...")

            cur.execute("""
                TRUNCATE TABLE
                    solubility.molecular_descriptors,
                    solubility.solubility_measurements,
                    solubility.compounds
                RESTART IDENTITY CASCADE;
            """)

            # -------------------------------------------------
            # 6. Prepare compound records
            # -------------------------------------------------

            compound_records = list(
                df[
                    [
                        "ID",
                        "Name",
                        "InChI",
                        "InChIKey",
                        "SMILES"
                    ]
                ].itertuples(index=False, name=None)
            )

            # -------------------------------------------------
            # 7. Insert compounds
            # -------------------------------------------------

            print("\nLoading compounds...")

            execute_values(
                cur,
                """
                INSERT INTO solubility.compounds
                    (
                        compound_id,
                        name,
                        inchi,
                        inchikey,
                        smiles
                    )
                VALUES %s
                """,
                compound_records,
                page_size=1000
            )

            print(f"Loaded compounds: {len(compound_records):,}")

            # -------------------------------------------------
            # 8. Prepare solubility records
            # -------------------------------------------------

            measurement_records = list(
                df[
                    [
                        "ID",
                        "Solubility",
                        "SD",
                        "Ocurrences",
                        "Group"
                    ]
                ].itertuples(index=False, name=None)
            )

            # -------------------------------------------------
            # 9. Insert solubility measurements
            # -------------------------------------------------

            print("Loading solubility measurements...")

            execute_values(
                cur,
                """
                INSERT INTO solubility.solubility_measurements
                    (
                        compound_id,
                        solubility_logs,
                        standard_deviation,
                        occurrences,
                        group_label
                    )
                VALUES %s
                """,
                measurement_records,
                page_size=1000
            )

            print(
                f"Loaded measurements: "
                f"{len(measurement_records):,}"
            )

            # -------------------------------------------------
            # 10. Prepare descriptor records
            # -------------------------------------------------

            descriptor_columns = [
                "ID",
                "MolWt",
                "MolLogP",
                "MolMR",
                "HeavyAtomCount",
                "NumHAcceptors",
                "NumHDonors",
                "NumHeteroatoms",
                "NumRotatableBonds",
                "NumValenceElectrons",
                "NumAromaticRings",
                "NumSaturatedRings",
                "NumAliphaticRings",
                "RingCount",
                "TPSA",
                "LabuteASA",
                "BalabanJ",
                "BertzCT"
            ]

            descriptor_records = list(
                df[descriptor_columns]
                .itertuples(index=False, name=None)
            )

            # -------------------------------------------------
            # 11. Insert descriptors
            # -------------------------------------------------

            print("Loading molecular descriptors...")

            execute_values(
                cur,
                """
                INSERT INTO solubility.molecular_descriptors
                    (
                        compound_id,
                        mol_wt,
                        mol_logp,
                        mol_mr,
                        heavy_atom_count,
                        num_h_acceptors,
                        num_h_donors,
                        num_heteroatoms,
                        num_rotatable_bonds,
                        num_valence_electrons,
                        num_aromatic_rings,
                        num_saturated_rings,
                        num_aliphatic_rings,
                        ring_count,
                        tpsa,
                        labute_asa,
                        balaban_j,
                        bertz_ct
                    )
                VALUES %s
                """,
                descriptor_records,
                page_size=1000
            )

            print(
                f"Loaded descriptors: "
                f"{len(descriptor_records):,}"
            )

        # -----------------------------------------------------
        # 12. Commit transaction
        # -----------------------------------------------------

        conn.commit()

        print("\n" + "=" * 70)
        print("LOAD COMPLETE")
        print("=" * 70)

        print(f"Compounds:             {len(compound_records):,}")
        print(f"Measurements:          {len(measurement_records):,}")
        print(f"Molecular descriptors: {len(descriptor_records):,}")

    except Exception:

        conn.rollback()
        print("\nERROR: Transaction rolled back.")

        raise

    finally:

        conn.close()

        print("\nPostgreSQL connection closed.")


if __name__ == "__main__":
    load_data()
