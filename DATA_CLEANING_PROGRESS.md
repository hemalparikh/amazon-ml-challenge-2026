# Data Cleaning & Normalization Progress

## Completed

- Located and extracted the competition dataset.
- Inspected Source 1, Source 2, Source 3 and ground truth.
- Confirmed that one Source 1 entity can match multiple Source 2/Source 3 records.
- Inspected business name variations.
- Inspected business address variations.
- Tested Unicode normalization.
- Confirmed that Unicode combining marks must be preserved.
- Inspected legal suffix variations such as Inc, LLC, Ltd, LLP, Pvt and Private Limited.
- Inspected address abbreviations such as Rd, St, Ave, Dr, Blvd, Hwy and Ln.
- Inspected website/domain-style business names.
- Original competition TSV files have not been modified.

## Initial Safe Normalization Rules

### Business Name
- Convert to lowercase.
- Trim leading and trailing spaces.
- Preserve Unicode letters, combining marks and numbers.
- Remove punctuation/symbols.
- Collapse repeated spaces.

### Business Address
- Convert to lowercase.
- Trim leading and trailing spaces.
- Preserve Unicode letters, combining marks and numbers.
- Remove punctuation/symbols.
- Collapse repeated spaces.
- Preserve original address order.
- Do not automatically spell-correct.
- Do not automatically reorder address components.
- Missing address values are converted to an empty string instead of the text "nan".
- In the first 10,000 Source 2 records, normalized addresses showed very few actual duplicates, so duplicate addresses should not be assumed to identify the same business.

## Rules Not Yet Applied

- Legal suffix removal or standardization.
- Address abbreviation expansion.
- Automatic spelling correction.
- Website/domain removal.
- Business-name word removal.
- Address component reordering.

These rules will be tested against the training data before implementation.