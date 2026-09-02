CREATE SCHEMA IF NOT EXISTS solubility;

CREATE TABLE solubility.compounds (
    compound_id VARCHAR(50) NOT NULL,
    name TEXT,
    inchi TEXT,
    inchikey VARCHAR(50),
    smiles TEXT NOT NULL,

    CONSTRAINT compounds_pkey
        PRIMARY KEY (compound_id),

    CONSTRAINT compounds_inchikey_key
        UNIQUE (inchikey)
);

CREATE TABLE solubility.molecular_descriptors (
    descriptor_id SERIAL PRIMARY KEY,
    compound_id VARCHAR(50) NOT NULL,

    mol_wt DOUBLE PRECISION,
    mol_logp DOUBLE PRECISION,
    mol_mr DOUBLE PRECISION,
    heavy_atom_count INTEGER,
    num_h_acceptors INTEGER,
    num_h_donors INTEGER,
    num_heteroatoms INTEGER,
    num_rotatable_bonds INTEGER,
    num_valence_electrons INTEGER,
    num_aromatic_rings INTEGER,
    num_saturated_rings INTEGER,
    num_aliphatic_rings INTEGER,
    ring_count INTEGER,
    tpsa DOUBLE PRECISION,
    labute_asa DOUBLE PRECISION,
    balaban_j DOUBLE PRECISION,
    bertz_ct DOUBLE PRECISION,

    CONSTRAINT molecular_descriptors_compound_id_key
        UNIQUE (compound_id),

    CONSTRAINT molecular_descriptors_compound_id_fkey
        FOREIGN KEY (compound_id)
        REFERENCES solubility.compounds(compound_id)
);

CREATE TABLE solubility.solubility_measurements (
    measurement_id SERIAL PRIMARY KEY,
    compound_id VARCHAR(50) NOT NULL,

    solubility_logs DOUBLE PRECISION NOT NULL,
    standard_deviation DOUBLE PRECISION,
    occurrences INTEGER,
    group_label VARCHAR(10),

    CONSTRAINT solubility_measurements_compound_id_fkey
        FOREIGN KEY (compound_id)
        REFERENCES solubility.compounds(compound_id)
);