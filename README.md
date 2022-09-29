# CLSTM | Causality-Structured LSTM
[Lu Li](https://www.researchgate.net/profile/Lu-Li-69?ev=hdr_xprf)

### Introduction

we presented a new causality structure-based LSTM model (CLSTM), which could learn time interdependency and causality information for hydrometeorological applications. We applied and compared LSTM and CLSTM methods for estimating SM across 64 FLUXNET sites globally. The results showed that CLSTM dramatically increased the predictive performance compared with LSTM. The Nash-Sutcliffe Efficiency (NSE) suggested that more than 67% of sites witnessed an improvement of SM simulation larger than 10%.

### Edition

CLSTM has two editions. These two editions have some differences on causality test and V2 always perform better than V1 according our preliminary test on soil moisture forecasting. For CLSTM [V1](https://github.com/leelew/CLSTM/tree/main/v1), the causal relations are calculated based on linear correlation/linear and non-linear Granger causality test. For CLSTM [V2](https://github.com/leelew/CLSTM/tree/main/v2), the PGM are calculated by [PCMCI](https://github.com/jakobrunge) tests and we grouped the input features with the same causal windows. For each group, we generate CLSTM and give ensemble mean forecast.

### Notation

We emphasized two points on CLSTM avoid to mislead readers:
1) Causality-structure is not "true" causality but a "statistic" causality. 
2) The improvement of CLSTM may NOT caused by causality information. We think it caused by the exchanged information of deep & shallow features (like predRNN vs ConvLSTM).

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
