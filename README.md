# Credit Risk Classification with LogisticRegression and RandomOverSampler

This application was written to explore the sklearn machine learning library and how to work with imbalanced data sets, particularly with credit/loan data. Loan data are inherently imbalanced since healthy loans far outweigh risky loans. Therefore, using Logistic Regression, we compare the performance of the algorithm with the original data and the RandomOverSampled data.  

---

## Technologies 

This application runs on Python 3.7.

---

## Libraries

Import statements already present in program, refer to them below:

```python
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from imblearn.metrics import classification_report_imbalanced

import warnings
warnings.filterwarnings('ignore')
```

---

## Database

This program couldn't run without the CSV data:

* lending_data.csv

---

## Contributors

Thank you for Eric Cardena for teaching Rice's FinTech Boot Camp. He was instrumental in teaching and helping us understand this material. Thank you for Rice for preparing this curriculum and the corresponding data sets that are required to be used. 

**Rishi Prasadha**

**LinkedIn**: https://www.linkedin.com/in/rishi-prasadha-912212133/

**Instagram**: @therishiprasadha

**Twitter**: @RishiPrasadha

---

## Licenses 

MIT