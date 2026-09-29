from normalization import (
    normalize_name,
    normalize_address,
    normalize_country,
)


examples = [
    "B+ Retail Inc",
    "LLC Moncada Léarning Center",
    "International South Consultants Private Ltd",
    "Moyna's Coffee",
    "Pvt. EFS Print Ventures Ltd.",
    "राम मार्केटिंग प्राइवेट लिमिटेड",
]

print("NAME NORMALIZATION")
print("=" * 60)

for value in examples:
    print(f"Original : {value}")
    print(f"Normalized: {normalize_name(value)}")
    print()


address_examples = [
    "1795 Westchester Drive, High Point, NC",
    "2100 Cameron Drive,Unit APARTMENT G, Dundalk, MD",
    "Door No 183, 41St Cross, 22Nd Main 9Th Block Jayanagar, Bengaluru Urban, Bangalore, ಕರ್ನಾಟಕ",
]

print("ADDRESS NORMALIZATION")
print("=" * 60)

for value in address_examples:
    print(f"Original : {value}")
    print(f"Normalized: {normalize_address(value)}")
    print()


country_examples = [
    "US",
    "India",
    "France",
    "FRANCE",
]

print("COUNTRY NORMALIZATION")
print("=" * 60)

for value in country_examples:
    print(f"Original : {value}")
    print(f"Normalized: {normalize_country(value)}")
print("COMPACT REPRESENTATIONS")
print("=" * 60)

test_names = [
    "ABC Technologies",
    "ABCTechnologies",
    "Moyna's Coffee",
]

for value in test_names:
    normalized = normalize_name(value)
    compact = normalized.replace(" ", "")

    print(f"Original   : {value}")
    print(f"Normalized : {normalized}")
    print(f"Compact    : {compact}")
    print()