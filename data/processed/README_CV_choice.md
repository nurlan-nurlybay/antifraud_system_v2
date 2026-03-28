# README_CV_choice

We don’t do nested K-fold CV even for non sequential models like MLP because it makes biased validation.

say 10 outer folds and 10 inner folds:

- [10] [90] - you test on 10 and train on 90
- [90] → [9][9][9]… →[9] [81] - you train on 81 and validate / optimize for 9

And that goes for each inner fold, then you have 10 good inner models, you can take the best one or the average of top 3 and fit the model to the whole inner data (90) and test on the outer 10. Such for each outer 10, so you have 10 good outer models. They give you two benefits:

1. With them you can make predictions for 100% of the dataset because for each outer fold there is a good model that hasn’t seen it.
2. You can take the best model or the average of top 3 to fit on the whole dataset.

## Why it doesn’t work for fraud

Frauds come in trends / schemes that are effective for a certain period before they get patched. So there may be a lot of type A frauds in January, type B frauds in February, etc. In k-fold CV, you train most models (validation fold in the middle) on a lot of neighboring data.

Let’s say it takes 3 months for a specific fraud trend to completely fade off and only after that the type A fraud only shares the common universal fraud characteristics with other frauds. You have 10 inner folds. When training models 4, 5, 6, and 7, 6/9 folds are biased. 3 months prior and 3 months subsequent.

So these three models are going to look at the training models and pick the sets of hyperparameters that overfit for those temporary fraud trends / characteristics because they share the same characteristics instead of the parameters that use rigorous regularization to avoid overfitting for temporary strong signals and focus more on universal fraud characteristics.

Models 1 and 10 are the least biased and 2, 3, 8, 9 are moderately biased. 2 minimal biased, 2 a little biased, 2 notably biased, and 4 maximum biased models. Naturally, the maximum biased models are going to demonstrate high AUC scores so you are going to choose among them so all of your 10 outer “good” models will be maximum biased.

## Made up bias metric

- for each outer model you have 10 inner folds/models.
- 4 of them are 6/9 biased, 2 of them are 5/9 biased, 2 of them are 4/9 biased, and 2 of them are 3/9 biased.
- Total = 49 bias for 90 training folds. 54%.

## Solution: walk-forward CV

You don’t have inner or outer folds you just have 10 folds.
You train on the first 5 and validate on the 6th, train on the first 6 and validate on the 7th etc. Train 0-i, validate i+1, you can also test on i+2 to get unseen predictions for most of the dataset.

So you have models for folds 6, 7, 8, 9, 10.
Bias socre = 3/5, 3/6, 3/7, 3/8, 3/9 = 15 biased folds for 35 total folds = 42%.

But here the benefit is not just 12%, this is a made up metric just for intuition and benefit is not linear.

In the k-fold CV, all of the models are biased to the maximum (67% or 6/9) if you choose the best model for each outer fold or the average of top 3.

### k-fold CV

- top 1 bias = 67%
- top 3 bias = 67%
- top 10 (average of all) bias = 54%
- worst 1 = worst 2 bias = 33%

### walk-forward CV

- top 1 bias = 33%
- top 3 bias = 38%
- average bias (depending on where you start could be a little higher) = 42%
- worst bias (depending on where you start can be 100%) = 60%

So obviously you are going to pick the best model which has half the bias of k-fold CV. Which is btw the same as picking the worst k-fold CV model, but why would you do that? Tt gets the same model, yes, but it is computationally more expensive quadratically vs linear and completely counter-intuitive and unorthodox.

The other pushback is, in walk-forward CV, the later models have less trend bias, so why not just do the naive 90 / 10 split, train the single best model instead of 6 or more questionable. Answer: same reason you k-fold CV for non-sequential data instead of the naive split. The last 10% could just be lucky and biased by chance, not only by trend or context. So you want to look at more models even if they are more contextually biased, that’s just the trade off.

Then you look at them and see which hyperparameters consistently performed good and you assemble your final model. When deciding on the final hyperparameters, you’ll have
models: 1, 2, 3, 4, 5, 6, …
These are sorted by how biased they are in an ascending oreder. First of all, you can limit the number of models: start with train on folds 1-6 not 1-4 for example so that the most biased models are ignored. Secondly, you can give more weights to less contextually biased models when deciding on the best model. So you don’t take the average model, you take the weighted average model. You can do it mathematically or just approximately / intuitively.