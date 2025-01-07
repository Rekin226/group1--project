# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 12:25:24 2024

@author: User
"""

import pandas as pd
df = pd.DataFrame({
    "topic": ["Plant Care"],
    "subtopic": ["Plant nutrients"],
    "content": ["To improve photosynthesis, nutrient levels, and growth while identifying system limitations."],
    "tags": ["gas exchange, light response curves"],
    "source": ["https://dx.doi.org/10.3390/horticulturae9030291"]
})
print(df["source"])








#df = pd.DataFrame({
#    "topic": ["Fish Nutrition", "Plant Nutrition", "Water Quality"],
#    "subtopic": ["Fish nutrients", "Plant growth", "Water pH"],
#    "content": [
#        "Potassium, iron, copper accumulation details.",
#        "Nutrient absorption and photosynthesis.",
#        "Optimal pH range for aquaculture."
#    ],
#    "tags": ["Fish health", "Plant care", "Water quality"]
#})

#data = {
#    "topic": "Fish Nutrition",
#    "subtopic": "Fish feed",
#   "content": "Using outputs for biogas and fertilizer in aquaponics with added nutrients is feasible.",
#    "tags": "Fish feed,Fish health"
#}

#new_data = pd.DataFrame([data])
#df = pd.concat([df, new_data], ignore_index=True)
#tagged_data = df[df["tags"].str.contains("Fish feed")]
#print(tagged_data)
