import re
import unicodedata
import pandas as pd


# Common legal/business suffixes.
# These are removed only from normalized business names.
LEGAL_SUFFIXES = {
    "inc",
    "incorporated",
    "corp",
    "corporation",
    "co",
    "company",
    "ltd",
    "limited",
    "llc",
    "llp",
    "plc",
    "pvt",
    "private",
    "proprietary",
    "pte",
}


def normalize_text(value):
    """
    General text normalization.

    - Handles missing values
    - Preserves Unicode scripts
    - Case-folds text
    - Removes punctuation safely
    - Normalizes whitespace
    """

    if pd.isna(value):
        return ""

    value = str(value).strip()

    if not value:
        return ""

    # Unicode normalization.
    value = unicodedata.normalize("NFKC", value)

    # Lowercase/case-insensitive normalization.
    value = value.casefold()

    # Remove apostrophes without creating an extra token.
    value = re.sub(r"['’]", "", value)

    # Replace punctuation with spaces.
    # Keep Unicode letters, numbers and combining marks.
    cleaned = []

    for char in value:
        category = unicodedata.category(char)

        if char.isspace():
            cleaned.append(" ")
        elif category.startswith(("L", "N", "M")):
            cleaned.append(char)
        else:
            cleaned.append(" ")

    value = "".join(cleaned)

    # Normalize whitespace.
    value = re.sub(r"\s+", " ", value).strip()

    return value


def normalize_name(value):
    """
    Normalize business names.

    Keeps the original language/script.
    Removes common legal suffixes from the end.
    """

    value = normalize_text(value)

    if not value:
        return ""

    tokens = value.split()

    # Remove common English legal suffixes from the end.
    while tokens and tokens[-1] in LEGAL_SUFFIXES:
        tokens.pop()

    return " ".join(tokens)


def normalize_address(value):
    """
    Normalize business addresses.

    Keeps numbers, words and non-Latin scripts because
    they can be important for entity resolution.
    """

    value = normalize_text(value)

    if not value:
        return ""

    # Normalize spacing around slash.
    value = re.sub(r"\s*/\s*", " / ", value)

    # Normalize spacing around hyphen.
    value = re.sub(r"\s*-\s*", "-", value)

    return value.strip()


def normalize_country(value):
    """
    Normalize country labels without restricting the
    allowed countries to a fixed list.
    """

    return normalize_text(value)


def normalize_dataframe(df):
    """
    Add normalized columns while preserving all original columns.
    """

    result = df.copy()

    result["normalized_name"] = result["business_name"].map(
        normalize_name
    )

    result["normalized_address"] = result["business_address"].map(
        normalize_address
    )

    result["normalized_country"] = result["country"].map(
        normalize_country
    )

    # Compact representations.
    # These are additional signals and do not replace
    # the space-preserving normalized fields.
    result["name_compact"] = result["normalized_name"].str.replace(
        " ", "", regex=False
    )

    result["address_compact"] = result["normalized_address"].str.replace(
        " ", "", regex=False
    )

    return result