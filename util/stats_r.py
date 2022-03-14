"""This file contains functions to perfrom hypothesis testing using R."""
import rpy2.robjects as robjects

wilcox_r = robjects.r['wilcox.test']
shapiro_r = robjects.r['shapiro.test']
numeric = robjects.r['as.numeric']
t_r = robjects.r['t.test']


def wilcox(x, y, alternative="two.sided"):
    """
    Return wilcox of x and y.

    Source: https://www.rdocumentation.org/packages/stats/versions/3.6.1/topics/wilcox.test
    :param x,y
    :param alternative: "two.sided", "less", "greater"
    """
    if all(r==x[0] for r in x) or all(r==y[0] for r in y):
        return 1
    x_num = numeric(x)
    y_num = numeric(y)
    return wilcox_r(x_num, y_num, alternative=alternative).rx("p.value")[0][0]


def t_test(x, y, alternative="two.sided"):
    """
    Return wilcox of x and y.

    Source https://www.rdocumentation.org/packages/stats/versions/3.6.1/topics/wilcox.test
    :param x,y
    :param alternative: "two.sided", "less", "greater"
    """
    x_num = numeric(x)
    y_num = numeric(y)
    return t_r(x_num, y_num, alternative=alternative).rx("p.value")[0][0]
