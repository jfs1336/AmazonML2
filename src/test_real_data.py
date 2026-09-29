from data_loader import load_training_data
from normalization import normalize_dataframe


print("Loading training data...")

source1, source2, source3, ground_truth = load_training_data()

print("Data loaded.")
print()


# Take a small sample first.
sample1 = source1.head(1000)
sample2 = source2.head(1000)
sample3 = source3.head(1000)


print("=" * 60)
print("NORMALIZING SAMPLE DATA")
print("=" * 60)

normalized1 = normalize_dataframe(sample1)
normalized2 = normalize_dataframe(sample2)
normalized3 = normalize_dataframe(sample3)


print("\nSource 1")
print(normalized1[
    [
        "entity_id",
        "business_name",
        "normalized_name",
        "business_address",
        "normalized_address",
        "country",
        "normalized_country",
    ]
].head(10).to_string(index=False))


print("\nSource 2")
print(normalized2[
    [
        "entity_id",
        "business_name",
        "normalized_name",
        "business_address",
        "normalized_address",
        "country",
        "normalized_country",
    ]
].head(10).to_string(index=False))


print("\nSource 3")
print(normalized3[
    [
        "entity_id",
        "business_name",
        "normalized_name",
        "business_address",
        "normalized_address",
        "country",
        "normalized_country",
    ]
].head(10).to_string(index=False))


print("\n" + "=" * 60)
print("NORMALIZATION CHECK")
print("=" * 60)

for name, df in [
    ("Source 1", normalized1),
    ("Source 2", normalized2),
    ("Source 3", normalized3),
]:
    print(f"\n{name}")
    print("Rows:", len(df))
    print("Normalized name column:", "normalized_name" in df.columns)
    print("Normalized address column:", "normalized_address" in df.columns)
    print("Normalized country column:", "normalized_country" in df.columns)

    print(
        "Empty normalized names:",
        (df["normalized_name"] == "").sum()
    )

    print(
        "Empty normalized addresses:",
        (df["normalized_address"] == "").sum()
    )

    print(
        "Empty normalized countries:",
        (df["normalized_country"] == "").sum()
    )