import ee
ee.Initialize(project='prescriptive-ai')
try:
    col = ee.ImageCollection("projects/sat-io/open-datasets/WorldPop/Global2")
    print("Found size:", col.size().getInfo())
except Exception as e:
    print("Error:", e)
