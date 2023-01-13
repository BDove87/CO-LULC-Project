# Colorado Land Use Land Cover (LULC) Prediction

This project uses eo-learn and Scikit-Learn to predict LULC via Sentinelhub imagery. The code generates eo-patches which are too large to upload to the github repository. The repository contains the final trained model as well as a number of graphs and geotiffs, the final prediction graph being the most useful. The geotiffs look black but information can be extracted from them.

This project is based off a tutorial from eo-learn (https://eo-learn.readthedocs.io/en/latest/examples/land-cover-map/SI_LULC_pipeline.html)

## Requirements

eo-learn
Reference data for training - this project was trained using data from the Denver Regional Council of Governments (https://drcog.org/services-and-resources/data-maps-and-modeling/regional-land-use-land-cover-project)