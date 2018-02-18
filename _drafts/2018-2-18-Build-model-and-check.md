---
layout: post
title: First Post

---

We left **Insert link to previous here** off from on the last post after performing an ANOVA statistical test and showing that these is a difference between treatments. 
Our next steps are to build a model and then check the model

<script type="text/javascript" async
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

# Outline

* Create Model
* Computing Residuals
* Checking for Outliers

# Model the data

At this stage, we have enough information to build a simple model. Remember from Part I that we were using an effects model with a single factor or treatment. 
The skeleton for this model is: 

$$ y_{i,j} = \mu + \tau_{i} + \epsilon_{ij} \space\space \lbrace\begin{array}{ll} i = 1, 2 ... a \\ j = 1 ,2 ... n \end{array} $$

From my previous post we had created a dataframe with our 



{% highlight python %}
## Point estimator of mu_i is mu_hat plus ti hat which is yi_bar
## Mu hat is also y hat
mu_hat = yi_avg
mu_hat
{% endhighlight %}

