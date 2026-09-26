import sys
sys.path.insert(0, "src")

import pandas as pd

from preprocessing.preprocess import preprocess_data


def test_preprocess_data():
    df = pd.DataFrame({
        "entity_id": ["S1-1"],
        "business_name": ["  B+ Retail Inc  "],
        "business_address": ["1712 Montebello AVE, Phoenix, AZ"],
        "country": ["US"],
    })

    result = preprocess_data(df)

    assert "business_name_normalized" in result.columns
    assert "business_address_normalized" in result.columns

    assert result.loc[0, "business_name"] == "  B+ Retail Inc  "
    assert result.loc[0, "business_name_normalized"] == "b retail inc"

    assert result.loc[0, "business_address"] == "1712 Montebello AVE, Phoenix, AZ"
    assert result.loc[0, "business_address_normalized"] == "1712 montebello avenue phoenix az"
