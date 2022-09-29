# CLSTM
Causality-Structured LSTM
[Lu Li](https://www.researchgate.net/profile/Lu-Li-69?ev=hdr_xprf)

### Intro
We emphasized that this causality is not "true" causality but a "statistic" causality. 

### Edition

CLSTM has two editions. These two editions have some differences and V2 always perform better than V1 according our preliminary test on soil moisture forecasting. We list the difference of two editions as following:

1) Different causality test
For CLSTM(V1), the causal relations are calculated based on linear correlation/linear and non-linear Granger causality test.
For CLSTM(V2), the causal relations are calculated by PCMCI tests, and generate 

### Citation

In case you use CLSTM in your research or work, please cite:

```bibtex
@article{Lu Li,
    author = {Lu Li, Yongjiu Dai et al.},
    title = {Causality-Structured Deep Learning for Soil Moisture Predictions},
    journal = {Journal of Hydrometeorlogy},
    year = {2022}
}
```

### [License](https://github.com/leelew/CLSTM/LICENSE)

Copyright (c) 2022, Lu Li
