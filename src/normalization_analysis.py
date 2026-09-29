from data_loader import load_training_data
from normalization import normalize_dataframe


print("Loading training data...")
source1, source2, source3, ground_truth = load_training_data()

print("Data loaded.")
print()


def analyze_source(name, df):
    print("=" * 70)
    print(name)
    print("=" * 70)

    normalized = normalize_dataframe(df)

    print("Rows:", len(normalized))

    print("\nMissing original values:")
    print("Business name:", normalized["business_name"].isna().sum())
    print("Business address:", normalized["business_address"].isna().sum())
    print("Country:", normalized["country"].isna().sum())

    print("\nEmpty normalized values:")
    print(
        "Normalized name:",
        (normalized["normalized_name"] == "").sum()
    )

    print(
        "Normalized address:",
        (normalized["normalized_address"] == "").sum()
    )

    print(
        "Normalized country:",
        (normalized["normalized_country"] == "").sum()
    )

    print("\nUnique normalized values:")

    print(
        "Normalized names:",
        normalized["normalized_name"].nunique()
    )

    print(
        "Normalized addresses:",
        normalized["normalized_address"].nunique()
    )

    print(
        "Normalized countries:",
        normalized["normalized_country"].nunique()
    )

    return normalized


source1_norm = analyze_source("SOURCE 1", source1)
source2_norm = analyze_source("SOURCE 2", source2)
source3_norm = analyze_source("SOURCE 3", source3)


print("\n")
print("=" * 70)
print("CROSS-SOURCE EXACT NORMALIZED MATCHES")
print("=" * 70)


# Ignore empty normalized values when calculating overlaps.
s1_names = set(
    source1_norm.loc[
        source1_norm["normalized_name"] != "",
        "normalized_name"
    ]
)

s2_names = set(
    source2_norm.loc[
        source2_norm["normalized_name"] != "",
        "normalized_name"
    ]
)

s3_names = set(
    source3_norm.loc[
        source3_norm["normalized_name"] != "",
        "normalized_name"
    ]
)


s1_addresses = set(
    source1_norm.loc[
        source1_norm["normalized_address"] != "",
        "normalized_address"
    ]
)

s2_addresses = set(
    source2_norm.loc[
        source2_norm["normalized_address"] != "",
        "normalized_address"
    ]
)

s3_addresses = set(
    source3_norm.loc[
        source3_norm["normalized_address"] != "",
        "normalized_address"
    ]
)


print("\nNormalized name overlap:")

print("S1 ∩ S2:", len(s1_names & s2_names))
print("S1 ∩ S3:", len(s1_names & s3_names))


print("\nNormalized address overlap:")

print("S1 ∩ S2:", len(s1_addresses & s2_addresses))
print("S1 ∩ S3:", len(s1_addresses & s3_addresses))


print("\n")
print("=" * 70)
print("EXACT NAME + ADDRESS + COUNTRY OVERLAP")
print("=" * 70)


s1_triples = set(
    zip(
        source1_norm["normalized_name"],
        source1_norm["normalized_address"],
        source1_norm["normalized_country"],
    )
)

s2_triples = set(
    zip(
        source2_norm["normalized_name"],
        source2_norm["normalized_address"],
        source2_norm["normalized_country"],
    )
)

s3_triples = set(
    zip(
        source3_norm["normalized_name"],
        source3_norm["normalized_address"],
        source3_norm["normalized_country"],
    )
)


print("S1 ∩ S2:", len(s1_triples & s2_triples))
print("S1 ∩ S3:", len(s1_triples & s3_triples))