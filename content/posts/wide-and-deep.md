---
title: Wide and Deep Learning 
date: 2024-12-29
---


Authors: Cheng et al (Google Inc)

Problem Statement: 

Learning user preferences in real world recommender systems is a mix of both memorization (nailing down past preferences) and generalization (using past data to recommend interesting items to users). While embedding based dense retrieval models contribute to the generalization aspect, they can sometimes be too general when the signals they have to learn from are sparse; for eg: a user item interaction matrix of the scale of Google Play Store is likely to be very sparse, where users may have interacted with only a minute percentage of the item inventory (in this case all the Play store apps). So a hybrid setting which can have both memorization and generalization is necessary to derive the best of both worlds. Not that this hybrid architecture is different from an ensemble, where models in the ensemble are trained separately. This paper proposes a joint optimization of the memorization and generalization components. 

An illustration of the Wide and Deep architecture 

Generalized Linear models are excellent architectures for memorization. Building and maintaining generalized linear models requires astute feature engineering, and a wide variety of cross feature interactions can be included as features to the GLM. These engineered features can also vary in granularity. Eg: (a bool feature to indicate if user has installed netflix, viewed paramount, if a user is both female and in India, etc). These features have several advantages in that they are interpretable, and can capture correlations between features and target labels very well. On the other hand, dense models involve projecting features to a lower dimensional embedding space and capturing more complex interactions. The disadvantage is that these representations are often not interpretable, and can lead to overgeneralization and poor recommendations at times. 

A Wide and Deep model involves training the wide part (the generalized linear model) and the Deep part (embedding model) in a joint setting. The wide part is optimized using FTRL (Follow the regularized leader) algorithm while deep part is optimized using AdaGrad.

A key point to note is that this architecture is a ranking model and not a retrieval model. The retrieval model is usually a faster, and smaller model which is optimized for recall. The Wide and Deep model is a ranking model that takes in some form of item representations, user representations, user-item interaction representations and solves for a logistic regression problem - how likely is the user to click/convert on this item ?

How the wide and deep features all come together

Cross Product Transformation:

This part is where the “Wide” part of the model comes from. This section is not explained in detail and I had to scourge through StackExchange a bit to really understand this better.  (This post was helpful). 

The following figure illustrates what cross product transformations mean 

For eg: if we consider only boolean flags like “impressed” and “installed”, we can make a cross interaction feature matrix as shown. For eg: the first row in the matrix indicates user has thus far installed A, C after viewing B while not installing the others. This kind of feature interactions and flags is a crucial part of the “memorization” aspect of the model and succinctly capture historical user interactions. 

The wide part is formed by concatenating the features (manually created) and the cross product transformations and projecting them into a single scalar using a linear transformation. The dense representation is similarly projected onto a single scalar. The features from wide and dense models (along with bias) are transformed into the {0,1} space through a sigmoid layer, and a logistic loss is used to optimize the model. 

Generating predictions from the Wide and deep model 

Note that the prediction is the user acquisition probability, which is essentially how likely user would be to install a certain app. 

Optimizing the wide and deep components

The wide part of the model is optimized using an algorithm called FTRL (follow the regularized model), while the dense part is optimized using Adagrad, an optimizer that is routinely used in deep learning training. In this post, we will look briefly into the FTRL algorithm, and the rationale for using this to optimize the wide part. 


Resources: 

https://datascience.stackexchange.com/questions/58907/understanding-the-wide-part-of-googles-wide-and-deep

https://arxiv.org/pdf/1606.07792

