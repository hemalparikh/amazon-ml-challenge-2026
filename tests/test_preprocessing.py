import sys

sys.path.insert(0, "src")

from preprocessing.name_normalizer import normalize_business_name
from preprocessing.address_normalizer import normalize_business_address


def test_name_normalization():
    assert normalize_business_name(
        "  Orelee's Barbershop  "
    ) == "orelee s barbershop"

    assert normalize_business_name(
        "B+ Retail Inc"
    ) == "b retail inc"

    assert normalize_business_name(
        "Dréxkor"
    ) == "dréxkor"

    assert normalize_business_name(
        "राम मार्केटिंग प्राइवेट लिमिटेड"
    ) == "राम मार्केटिंग प्राइवेट लिमिटेड"


def test_address_normalization():
    assert normalize_business_address(
        "1795 Westchester Drive, High Point, NC"
    ) == "1795 westchester drive high point nc"

    assert normalize_business_address(
        "2062 Smith Hollow RD, Blaine, TN"
    ) == "2062 smith hollow road blaine tn"

    assert normalize_business_address(
        "914 Pierpont AVE, Cleveland, OH"
    ) == "914 pierpont avenue cleveland oh"

    assert normalize_business_address(
        "1418 Meadowbrook DR, Johnson City, TN"
    ) == "1418 meadowbrook drive johnson city tn"

    assert normalize_business_address("   ") == ""

    assert normalize_business_address(None) == ""

    assert normalize_business_address(float("nan")) == ""

