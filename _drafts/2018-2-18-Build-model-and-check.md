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

and we can group $$ \mu_{i}= \mu + \tau_{i} $$ and this essentially tells us that $\mu_{i} is a sum of $$\mu$$ and $$\tau_{i}$$.

From my previous post we had created a dataframe, yi_avg, which was the average under the i-th treatment. It turns out that this average is also a good estimate for our point estimator, $$ \hat{mu}_{i}$$. 
Tip: Anything you see with a hat over a variable means that it is an estimator.
We can now create our model:

{% highlight python %}
## Point estimator of mu_i is mu_hat plus ti hat which is yi_bar
## Mu hat is also y hat
y_hat = yi_avg 
y_hat
{% endhighlight %}

![Check out y_hat](/images/Build-model-and-check/createPountEstimatorMuHat.png){:class="img-responsive"}

Great! Now we have a simple model to predict our outcome variable, y (CVD growth rate). 

# Computing Residuals

To calculate our residuals, we need to subtract our model from our actual values. 

{% highlight python %}
## Point estimator of mu_i is mu_hat plus ti hat which is yi_bar
## Mu hat is also y hat
cvd_pivot_residuals = cvd_pivot.sub(y_hat, axis = 1)
cvd_pivot_residuals
{% endhighlight %}

![Check out residuals](/images/Build-model-and-check/cvdPivotResiduals.png){:class="img-responsive"}

And melt them back into a singular column for analysis with Python.

{% highlight python %}
## Point estimator of mu_i is mu_hat plus ti hat which is yi_bar
## Mu hat is also y hat
cvd_pivot_residuals_melted = cvd_pivot_residuals.melt()
cvd_pivot_residuals_melted
{% endhighlight %}

![Check out residuals](/images/Build-model-and-check/cvdPivotResidualsMelted.png){:class="img-responsive"}


# Testing assumptions with Residuals

From here we need to check the assumptions we originally made.  We do that by:

* Checking normality against residuals
* Plot versus time sequence
* Plot versus fitted values.
* Check for outliers