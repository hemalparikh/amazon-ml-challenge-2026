import sys

sys.path.insert(0, "src")

from preprocessing.name_normalizer import normalize_business_name
from preprocessing.address_normalizer import normalize_business_address


def test_name_normalization():
    assert normalize_business_name("  Orelee's Barbershop  ") == "orelee s barbershop"
    assert normalize_business_name("B+ Retail Inc") == "b retail inc"
    assert normalize_business_name("Dréxkor") == "dréxkor"
    assert normalize_business_name("राम मार्केटिंग प्राइवेट लिमिटेड") == "राम मार्केटिंग प्राइवेट लिमिटेड"


def test_address_normalization():
    assert normalize_business_address(
        "1795 Westchester Drive, High Point, NC"
    ) == "1795 westchester drive high point nc"

    assert normalize_business_address("   ") == ""
    assert normalize_business_address(None) == ""