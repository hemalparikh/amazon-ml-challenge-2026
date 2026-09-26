import sys
sys.path.insert(0, "src")

from preprocessing.name_normalizer import normalize_business_name


def test_multilingual_names():
    examples = [
        "राम मार्केटिंग प्राइवेट लिमिटेड",
        "சதர்ன் இண்டஸ்ட்ரீஸ் பிரைவேட் லிமிடெட்",
        "શક્તિ અર્બન પ્રોડક્ટ્સ પ્રાઇவேટ લિમિટેડ",
        "ಕರ್ನಾಟಕ",
        "Dréxkor",
    ]

    for name in examples:
        assert normalize_business_name(name) == name.lower()
