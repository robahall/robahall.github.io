---
layout: post
title: Part II Simple ANOVA - Build our model and check our assumptions 

---

We left off from the [last post](http://www.robahall.com/Performing-ANOVA-Analysis/) after performing an ANOVA statistical test and showing that there is a difference between treatments. 
Our next steps are to build a model and then check our model and assumptions we had previously made are correct. 

<script type="text/javascript" async
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

# Outline

* Model the data
* Computing residuals
* Testing assumptions with residuals
* Check for outliers

# Model the data

At this stage, we have enough information to build a simple model. Remember from Part I that we were using an effects model with a single factor or treatment. 
The skeleton for this model is: 

$$ y_{i,j} = \mu + \tau_{i} + \epsilon_{ij} \space\space \lbrace\begin{array}{ll} i = 1, 2 ... a \\ j = 1 ,2 ... n \end{array} $$

We can simplify $$\mu$$ and $$\tau_{i}$$ by taking $$ $$ \mu_{i}= \mu + \tau_{i} $$

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

# Computing residuals

To calculate our residuals, we need to subtract our model from our actual values. 

{% highlight python %}
cvd_pivot_residuals = cvd_pivot.sub(y_hat, axis = 1)
cvd_pivot_residuals
{% endhighlight %}

![Check out residuals](/images/Build-model-and-check/cvdPivotResiduals.png){:class="img-responsive"}

And melt them back into a singular column for analysis with Python.

{% highlight python %}
cvd_pivot_residuals_melted = cvd_pivot_residuals.melt()
cvd_pivot_residuals_melted
{% endhighlight %}

![Check out residuals](/images/Build-model-and-check/cvdPivotResidualsMelted.png){:class="img-responsive"}


# Testing assumptions with residuals

From here we need to check the assumptions we originally made.  We do that by:

* Checking normality against residuals
* Plot versus time sequence
* Plot versus fitted values.
* Check for outliers

One of the first assumptions we made when running an ANOVA was that the data was approximately normal. We typically check this with a histogram but because we have small sample small small fluctuations in our data causes large shape changes to our histogram. 
Instead we check our residuals with a normal probability plot. What we are interested in from our probability plot is whether our residuals are approximately normal. 

{% highlight python %}
from scipy import stats
stats.probplot(cvd_pivot_residuals_melted['value'], plot=plt)
{% endhighlight %}

![Check out probability plot](/images/Build-model-and-check/probPlotResiduals.png){:class="img-responsive"}

The important thing to look for is if the data between -1 and +1 quantiles are approximately linear. From the probability plot we see that our data is approximately normal with some deviations on the tails. 

At this point we would check to see if there is correlation between samples run in time sequence. If there was, then our assumption of independence between runs is false. Unfortunately since the data is generated within Python we will not be calculating it here. 

The next check is to make sure that there is consistent variance between treatments. An unequal variance can cause deviations in our F-Test but will mainly affect experiments with unequal treatment sizes or a different type of model. 

To set this up we have to reset our power settings to our power average value.  

{% highlight python %}
for power_setting, power_avg in mu_hat.iteritems():
    cvd_pivot_residuals_melted.loc[cvd_pivot_residuals_melted['Power'] == power_setting, 'Power'] = power_avg

{% endhighlight %}

![Set up predicted vs. resiudals](/images/Build-model-and-check/predictedVsResiduals.png){:class="img-responsive"}

{% highlight python %}
plt.scatter(cvd_pivot_residuals_melted['Power'], cvd_pivot_residuals_melted['value'])
plt.xlabel('Predicted Growth Rate')
plt.ylabel('Residuals')
plt.show()
{% endhighlight %}

![Plot predicted vs. resiudals](/images/Build-model-and-check/predictedGrowthVsResiduals.png){:class="img-responsive"}

So with this chart we can now scan the data and see if we have constant variance. We see that there is some differences in variance. With our balanced designed (same number of samples per treatment) this does not cause a large effect on our F test.
With the last few tests showing some deviations from normality and a deviation from homogeneous variance it stresses the importance of having a balanced experiment and how having a balanced experiment helps minimize deviations so we can accurately interpret the experiment. 

# Check for outliers

Looking at our data in the predicted versus our residual chart we see some large deviations from set to set. Lets check and see if they are outliers. 
A simple check is by checking the standardized residuals:

$$ d_{ij} = \frac{e_ij}{\sqrt{MSE}} $$

where $$e_ij$$ is our residual, $$MSE$$ is our mean squared error calculated previously. This equation normalizes our residuals and most of the data (~95%) should fall within $$\pm$$2 from our mean. Anything greater than 3 is a potential outlier.

{% highlight python %}
def check_outliers(residuals, mean_square_error):
    check_out = residuals.iloc[:,1:].divide(mean_square_error ** (0.5))
    return check_out[abs(check_out) > 1]
    
check_outliers(cvd_pivot_residuals_melted, MSE)
{% endhighlight %}

![Plot predicted vs. resiudals](/images/Build-model-and-check/outliers.png){:class="img-responsive"}

From this data we see that all of our data falls within 95% of our normal distribution of data and gives us no cause for concer. 


# Conclusions

We have successfully created a model and checked that we did not violate any of our assumptions we made in the previous post. We saw that our data was approximately normal, but did have some non-homogenous variance. 
Because we have a balanced data set we realized that our F test would not be dramatically affected. We also test our data for outliers and came to the conclusion that were none. 

