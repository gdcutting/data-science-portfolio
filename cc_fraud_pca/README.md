# Credit Card Fraud Detection

This sample project will use a variety of models to analyze the problem of credit card transaction fraud.

## Dataset

The dataset is available in more than one location. I'm posting these links because each site has a slightly different set of information about the dataset:
- [Datahub.io](https://datahub.io/machine-learning/creditcard)
- [OpenML.org](https://www.openml.org/search?type=data&sort=runs&id=1597&status=active))

> Note: the file for the credit card fraud data is ~144MB, which exceeds GitHub's 100MB limit. Therefore I am not including the file in the repo. If you want to download it, use either of the links above. For use with the Jupyter notebooks in this repo, rename the file 'creditcard.csv'

This dataset presents an interesting challenge because the data has already been processed by [Principal Component Analysis [PCA]](https://en.wikipedia.org/wiki/Principal_component_analysis), a common [dimensionality reduction](https://en.wikipedia.org/wiki/Dimensionality_reduction) technique which performs a linear transformation of the original into a new coordinate system where most of the variation in the data can be described with fewer dimensions than the original data. This method increases the interpretability of the data and makes visualization possible.

The application of PCA results in a dataset with a couple of dozen continuous numeric inputs that do not have an obvious intuitive meaning (like the distance between the transaction and the customer's home, as we would normally find in a credit card transaction dataset). So we will need to take a somewhat different approach to analyzing this data.

There are many different credit card fraud datasets available, and I am also working on a version of this project that uses more traditional fraud data that has not already been processed using PCA. Will begin posting those soon.