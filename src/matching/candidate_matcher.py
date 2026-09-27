from matching.matcher_normalizer import (
    normalize_match_name,
    normalize_match_address,
)


def normalize_country(country):
    if country is None:
        return ""
    return str(country).strip().lower()


def is_match(source1, candidate):
    name1 = normalize_match_name(source1.get("business_name"))
    name2 = normalize_match_name(candidate.get("business_name"))

    address1 = normalize_match_address(source1.get("business_address"))
    address2 = normalize_match_address(candidate.get("business_address"))

    country1 = normalize_country(source1.get("country"))
    country2 = normalize_country(candidate.get("country"))

    if not country1 or country1 != country2:
        return False

    if name1 and name2 and address1 and address2:
        return name1 == name2 and address1 == address2

    if name1 and name2 and not address1 and not address2:
        return name1 == name2

    if address1 and address2 and not name1 and not name2:
        return address1 == address2

    return False
def match_candidates(source1_row, candidate_rows):
    matched_ids = []

    for candidate in candidate_rows:
        if is_match(source1_row, candidate):
            matched_ids.append(candidate.get("entity_id"))

    return matched_ids

import csv


def read_candidate_pairs(path):
    with open(path, "r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file, delimiter="\t")
        for row in reader:
            yield row


def write_matching_results(path, results):
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, delimiter="\t")
        writer.writerow(["source1_entity_id", "matched_entity_ids"])

        for source1_id, matched_ids in results:
            writer.writerow([
                source1_id,
                ",".join(matched_ids),
            ])
def build_results(source1_rows, candidate_rows_by_id, candidate_pairs_rows):
    results = []

    for source1_id, source1_row in source1_rows.items():
        candidate_ids = candidate_pairs_rows.get(source1_id, [])

        candidates = [
            candidate_rows_by_id[candidate_id]
            for candidate_id in candidate_ids
            if candidate_id in candidate_rows_by_id
        ]

        matched_ids = match_candidates(source1_row, candidates)
        results.append((source1_id, matched_ids))

    return results

def load_candidate_pairs(path):
    pairs = {}

    for row in read_candidate_pairs(path):
        source1_id = row["source1_entity_id"]
        raw_candidate_ids = row.get("candidate_entity_ids", "")

        candidate_ids = [
            candidate_id.strip()
            for candidate_id in raw_candidate_ids.split(",")
            if candidate_id.strip()
        ]

        pairs[source1_id] = candidate_ids

    return pairs

def run_matching(source1_rows, candidate_rows_by_id, candidate_pairs_path, output_path):
    candidate_pairs_rows = load_candidate_pairs(candidate_pairs_path)

    results = build_results(
        source1_rows,
        candidate_rows_by_id,
        candidate_pairs_rows,
    )

    write_matching_results(output_path, results)
