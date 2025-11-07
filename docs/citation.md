(citation-doc)=

# Citation Guide and Bibliography

If you use NeMoS in a published academic article or presentation, please cite the code by the DOI. You can use the following:

- Code: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17553287.svg)](https://doi.org/10.5281/zenodo.17553287)

Additionally, it may be a good idea to cite the following paper(s) depending on which component you use:

- {class}`GLM <nemos.glm.GLM>` or {class}`PopulationGLM <nemos.glm.PopulationGLM>`: {cite:alp}`nelder1972generalized`.

- {class}`BSplineEval <nemos.basis.BSplineEval>` or {class}`BSplineConv <nemos.basis.BSplineConv>`: {cite:alp}`de1972calculating,cox1972numerical`.
- {class}`MSplineEval <nemos.basis.MSplineEval>` or {class}`MSplineConv <nemos.basis.MSplineConv>`: {cite:alp}`ramsay1988monotone`.
- {class}`RaisedCosineLinearEval <nemos.basis.RaisedCosineLinearEval>` or {class}`RaisedCosineLogEval <nemos.basis.RaisedCosineLogEval>` or {class}`RaisedCosineLinearConv <nemos.basis.RaisedCosineLinearConv>` or {class}`RaisedCosineLogConv <nemos.basis.RaisedCosineLogConv>`: {cite:alp}`pillow2005prediction`.

- {class}`ElasticNet <nemos.regularizer.ElasticNet>`: {cite:alp}`zou2005regularization`.
- {class}`GroupLasso <nemos.regularizer.GroupLasso>`: {cite:alp}`yuan2006model`.
- {class}`Lasso <nemos.regularizer.Lasso>`: {cite:alp}`tibshirani1996regression`.
- {class}`Ridge <nemos.regularizer.Ridge>`: {cite:alp}`hoerl1970ridge1,hoerl1970ridge2`.


## Bibliography

```{bibliography} references.bib
:style: plain
```
