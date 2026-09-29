from data_loader import load_training_data
from normalization import normalize_dataframe


print("Loading training data...")
source1, source2, source3, ground_truth = load_training_data()

print("Data loaded.")
print()


print("Normalizing sources...")

source1 = normalize_dataframe(source1)
source2 = normalize_dataframe(source2)
source3 = normalize_dataframe(source3)

print("Normalization complete.")
print()


# ---------------------------------------------------------
# Create lookup dictionaries from entity_id to normalized data
# ---------------------------------------------------------

s1_lookup = source1.set_index("entity_id")[
    [
        "normalized_name",
        "normalized_address",
        "normalized_country",
    ]
].to_dict("index")

s2_lookup = source2.set_index("entity_id")[
    [
        "normalized_name",
        "normalized_address",
        "normalized_country",
    ]
].to_dict("index")

s3_lookup = source3.set_index("entity_id")[
    [
        "normalized_name",
        "normalized_address",
        "normalized_country",
    ]
].to_dict("index")


# ---------------------------------------------------------
# Evaluate known ground-truth matches
# ---------------------------------------------------------

print("=" * 70)
print("GROUND-TRUTH NORMALIZATION ANALYSIS")
print("=" * 70)


total_matches = 0

name_matches = 0
address_matches = 0
country_matches = 0

name_address_matches = 0
name_country_matches = 0
address_country_matches = 0

all_three_matches = 0


for _, row in ground_truth.iterrows():

    s1_id = row["source1_entity_id"]
    matched_ids = row["matched_entity_ids"]

    # Empty ground-truth entries mean no known match.
    if not isinstance(matched_ids, str) or not matched_ids.strip():
        continue

    s1 = s1_lookup.get(s1_id)

    if s1 is None:
        continue

    matched_ids = matched_ids.split(",")

    for matched_id in matched_ids:

        matched_id = matched_id.strip()

        if matched_id.startswith("S2-"):
            candidate = s2_lookup.get(matched_id)
        elif matched_id.startswith("S3-"):
            candidate = s3_lookup.get(matched_id)
        else:
            continue

        if candidate is None:
            continue

        total_matches += 1

        same_name = (
            s1["normalized_name"] != ""
            and s1["normalized_name"] == candidate["normalized_name"]
        )

        same_address = (
            s1["normalized_address"] != ""
            and s1["normalized_address"] == candidate["normalized_address"]
        )

        same_country = (
            s1["normalized_country"] != ""
            and s1["normalized_country"] == candidate["normalized_country"]
        )

        if same_name:
            name_matches += 1

        if same_address:
            address_matches += 1

        if same_country:
            country_matches += 1

        if same_name and same_address:
            name_address_matches += 1

        if same_name and same_country:
            name_country_matches += 1

        if same_address and same_country:
            address_country_matches += 1

        if same_name and same_address and same_country:
            all_three_matches += 1


print("Total known matches:", total_matches)
print()


def percentage(value):
    if total_matches == 0:
        return 0.0

    return 100 * value / total_matches


print("Exact normalized agreement among TRUE matches")
print("-" * 70)

print(
    f"Name only:              {name_matches:,} "
    f"({percentage(name_matches):.2f}%)"
)

print(
    f"Address only:            {address_matches:,} "
    f"({percentage(address_matches):.2f}%)"
)

print(
    f"Country only:            {country_matches:,} "
    f"({percentage(country_matches):.2f}%)"
)

print(
    f"Name + Address:          {name_address_matches:,} "
    f"({percentage(name_address_matches):.2f}%)"
)

print(
    f"Name + Country:          {name_country_matches:,} "
    f"({percentage(name_country_matches):.2f}%)"
)

print(
    f"Address + Country:       {address_country_matches:,} "
    f"({percentage(address_country_matches):.2f}%)"
)

print(
    f"Name + Address + Country: {all_three_matches:,} "
    f"({percentage(all_three_matches):.2f}%)"
)