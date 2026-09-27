import regex

LEGAL_SUFFIXES = {
    "private",
    "limited",
    "corporation",
    "incorporated",
    "company",
    "llc",
    "plc",
    "pvt",
    "ltd",
}

ADDRESS_ABBREVIATIONS = {
    "rd": "road",
    "st": "street",
    "ave": "avenue",
    "av": "avenue",
    "dr": "drive",
    "blvd": "boulevard",
    "hwy": "highway",
    "ln": "lane",
    "pkwy": "parkway",
    "ct": "court",
    "pl": "place",
}

def normalize_match_name(name):
    if name is None:
        return ""

    name = str(name).lower().strip()
    name = name.replace("&", " and ")
    name = regex.sub(r"[^\p{L}\p{M}\p{N}\s]", " ", name)
    name = regex.sub(r"\s+", " ", name).strip()

    words = name.split()

    # Remove legal terms only when they appear at the end of the name.
    while words and words[-1] in LEGAL_SUFFIXES:
        words.pop()

    return " ".join(words)

def normalize_match_address(address):
    if address is None:
        return ""

    if regex.match(r"^\s*nan\s*$", str(address), regex.IGNORECASE):
        return ""

    address = str(address).lower().strip()
    address = regex.sub(r"[^\p{L}\p{M}\p{N}\s]", " ", address)

    words = []
    for word in address.split():
        words.append(ADDRESS_ABBREVIATIONS.get(word, word))

    return " ".join(words)
