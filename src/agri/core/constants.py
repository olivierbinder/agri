"""Domain reference data and defaults shared across jobs, the API, and the UI."""

# %% CROPS

# Optimal growing temperature (°C) used for feature engineering.
CROP_OPT_TEMPS: dict[str, float] = {
    "Wheat": 15,
    "Rice, paddy": 28,
    "Maize": 22,
    "Potatoes": 17,
    "Soybeans": 22,
    "Sorghum": 28,
    "Cassava": 27,
    "Sweet potatoes": 25,
    "Plantains and others": 26,
    "Yams": 25,
}

# All crops known to the model.
ITEMS: list[str] = sorted(CROP_OPT_TEMPS)

# Reference yield (hg/ha, median across all areas/years in yield_df.csv) used to
# normalize recommendations. Raw yields aren't comparable across crops (e.g. Potatoes
# is ~12x Soybeans regardless of climate), so ranking on absolute hg/ha always favors
# naturally high-yield crops. Dividing a crop's predicted yield by its own global
# reference gives a relative "how well does this crop do vs. its usual self" score.
# Computed globally (not per-Area) because most areas lack history for every crop.
CROP_REF_YIELD: dict[str, float] = {
    "Cassava": 128200.0,
    "Maize": 25401.0,
    "Plantains and others": 89860.5,
    "Potatoes": 182271.0,
    "Rice, paddy": 35878.0,
    "Sorghum": 12885.0,
    "Soybeans": 15533.0,
    "Sweet potatoes": 99940.0,
    "Wheat": 25497.0,
    "Yams": 92593.0,
}

# %% COUNTRIES

# All countries present in the training dataset (yield_df.csv).
AREAS: list[str] = [
    "Albania",
    "Algeria",
    "Angola",
    "Argentina",
    "Armenia",
    "Australia",
    "Austria",
    "Azerbaijan",
    "Bahamas",
    "Bahrain",
    "Bangladesh",
    "Belarus",
    "Belgium",
    "Botswana",
    "Brazil",
    "Bulgaria",
    "Burkina Faso",
    "Burundi",
    "Cameroon",
    "Canada",
    "Central African Republic",
    "Chile",
    "Colombia",
    "Croatia",
    "Denmark",
    "Dominican Republic",
    "Ecuador",
    "Egypt",
    "El Salvador",
    "Eritrea",
    "Estonia",
    "Finland",
    "France",
    "Germany",
    "Ghana",
    "Greece",
    "Guatemala",
    "Guinea",
    "Guyana",
    "Haiti",
    "Honduras",
    "Hungary",
    "India",
    "Indonesia",
    "Iraq",
    "Ireland",
    "Italy",
    "Jamaica",
    "Japan",
    "Kazakhstan",
    "Kenya",
    "Latvia",
    "Lebanon",
    "Lesotho",
    "Libya",
    "Lithuania",
    "Madagascar",
    "Malawi",
    "Malaysia",
    "Mali",
    "Mauritania",
    "Mauritius",
    "Mexico",
    "Montenegro",
    "Morocco",
    "Mozambique",
    "Namibia",
    "Nepal",
    "Netherlands",
    "New Zealand",
    "Nicaragua",
    "Niger",
    "Norway",
    "Pakistan",
    "Papua New Guinea",
    "Peru",
    "Poland",
    "Portugal",
    "Qatar",
    "Romania",
    "Rwanda",
    "Saudi Arabia",
    "Senegal",
    "Slovenia",
    "South Africa",
    "Spain",
    "Sri Lanka",
    "Sudan",
    "Suriname",
    "Sweden",
    "Switzerland",
    "Tajikistan",
    "Thailand",
    "Tunisia",
    "Turkey",
    "Uganda",
    "Ukraine",
    "United Kingdom",
    "Uruguay",
    "Zambia",
    "Zimbabwe",
]

# %% DEFAULTS

DEFAULT_AREA: str = "France"
DEFAULT_ITEM: str = "Wheat"
DEFAULT_YEAR: int = 2024
DEFAULT_RAINFALL: float = 800.0
DEFAULT_PESTICIDES: float = 150.0
DEFAULT_TEMP: float = 15.0

YIELD_UNIT: str = "hg/ha"
