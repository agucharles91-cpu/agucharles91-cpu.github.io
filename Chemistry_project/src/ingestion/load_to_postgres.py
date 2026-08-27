import pandas as pd
import psycopg2
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
CSV_PATH = PROJECT_ROOT / 'data' / 'raw' / 'curated-solubility-dataset.csv'

def load_data():
    print("LOADING DATA...")
    
    df = pd.read_csv(CSV_PATH)
    print(f"Rows: {len(df):,}")
    
    conn = psycopg2.connect(
        host='localhost',
        port=5432,
        database='chemical_solubility',
        user='postgres',
        password='REMOVED_FROM_HISTORY'
    )
    cur = conn.cursor()
    
    try:
        for i, row in df.iterrows():
            cur.execute("""
                INSERT INTO solubility.compounds (compound_id, name, inchi, inchikey, smiles)
                VALUES (%s, %s, %s, %s, %s);
            """, (row['ID'], row['Name'], row['InChI'], row['InChIKey'], row['SMILES']))
            
            cur.execute("""
                INSERT INTO solubility.solubility_measurements 
                (compound_id, solubility_logs, standard_deviation, occurrences, group_label)
                VALUES (%s, %s, %s, %s, %s);
            """, (row['ID'], row['Solubility'], row['SD'], row['Ocurrences'], row['Group']))
            
            cur.execute("""
                INSERT INTO solubility.molecular_descriptors (
                    compound_id, mol_wt, mol_logp, mol_mr, heavy_atom_count,
                    num_h_acceptors, num_h_donors, num_heteroatoms, num_rotatable_bonds,
                    num_valence_electrons, num_aromatic_rings, num_saturated_rings,
                    num_aliphatic_rings, ring_count, tpsa, labute_asa, balaban_j, bertz_ct
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
            """, (
                row['ID'], row['MolWt'], row['MolLogP'], row['MolMR'], row['HeavyAtomCount'],
                row['NumHAcceptors'], row['NumHDonors'], row['NumHeteroatoms'], row['NumRotatableBonds'],
                row['NumValenceElectrons'], row['NumAromaticRings'], row['NumSaturatedRings'],
                row['NumAliphaticRings'], row['RingCount'], row['TPSA'], row['LabuteASA'],
                row['BalabanJ'], row['BertzCT']
            ))
            
            if (i + 1) % 1000 == 0:
                print(f"Processed {i+1:,} rows")
        
        conn.commit()
        print(f"\n✓ LOADED {i+1:,} ROWS SUCCESSFULLY")
        
    except Exception as e:
        print(f"ERROR: {e}")
        conn.rollback()
    finally:
        cur.close()
        conn.close()

if __name__ == "__main__":
    load_data()