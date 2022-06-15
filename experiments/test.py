import pandas as pd

data = pd.DataFrame(columns=["Model", "Dataset"])
data = data.append([{"Model": "Resnet", "Dataset": "CUB200"}, {
                   "Model": "Resnet", "Dataset": "CUB200"}])
print(data)