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


def test_additional_address_abbreviations():
    assert normalize_business_address(
        "123 Main Pkwy, Dallas, TX"
    ) == "123 main parkway dallas tx"

    assert normalize_business_address(
        "45 Oak Ct, Boston, MA"
    ) == "45 oak court boston ma"

    assert normalize_business_address(
        "78 Market Pl, Mumbai, India"
    ) == "78 market place mumbai india"


def test_legal_business_name_normalization():
    assert normalize_business_name(
        "Zander Blue Company"
    ) == "zander blue co"

    assert normalize_business_name(
        "Vadodara Industries Pvt Limited"
    ) == "vadodara industries pvt ltd"

    assert normalize_business_name(
        "Libra Service Limited"
    ) == "libra service ltd"

    assert normalize_business_name(
        "Precision Staffing Corporation"
    ) == "precision staffing corp"

    assert normalize_business_name(
        "Red Ventures Private Limited"
    ) == "red ventures pvt ltd"

    assert normalize_business_name(
        "Million Marketing Pvt. Limited"
    ) == "million marketing pvt ltd"

    assert normalize_business_name(
        "Olszewski Holding Company LLC LLC"
    ) == "olszewski holding co llc llc"


def test_preprocess_missing_values():
    import numpy as np
    import pandas as pd
    from preprocessing.preprocess import preprocess_data

    df = pd.DataFrame({
        "business_name": ["", None, np.nan],
        "business_address": ["", None, np.nan],
        "country": ["", None, np.nan],
    })

    result = preprocess_data(df)

    assert result["business_name_normalized"].tolist() == ["", "", ""]
    assert result["business_address_normalized"].tolist() == ["", "", ""]
    assert result["country"].isna().tolist() == [False, True, True]


def test_multilingual_unicode_normalization():
    assert normalize_business_name(
        "\u0930\u093e\u092e \u092e\u093e\u0930\u094d\u0915\u0947\u091f"
    ) == "\u0930\u093e\u092e \u092e\u093e\u0930\u094d\u0915\u0947\u091f"

    assert normalize_business_name(
        "\u0ba8\u0bb2\u0bcd\u0bb2 \u0b95\u0b9f\u0bc8"
   ) == "\u0ba8\u0bb2\u0bcd\u0bb2 \u0b95\u0b9f\u0bc8"

    assert normalize_business_name(
        "Caf\u00e9 Milano"
    ) == "caf\u00e9 milano"


def test_address_numbers_and_postal_codes_are_preserved():
    assert normalize_business_address(
        "12-A, MG Road, Mumbai 400001"
    ) == "12 a mg road mumbai 400001"

    assert normalize_business_address(
        "45/2, Oak St., Pune 411001"
    ) == "45 2 oak street pune 411001"

    assert normalize_business_address(
        "Flat #7, 123 Main Rd, Delhi 110001"
    ) == "flat 7 123 main road delhi 110001"