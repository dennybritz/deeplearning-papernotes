## [Wide & Deep Learning for Recommender Systems](https://arxiv.org/abs/1606.07792)

TLDR; The authors jointly train a Logistic Regression Model with sparse features that is good at "memorization" and a deep feedforward net with embedded sparse features that is good at "generalization". The model is live in the Google Play store and has achieved a 3.9% gain in app acquisiton as measured by A/B testing.

#### Key Points

- Wide Model (Logistic Regression) gets cross product of binary features, e.g. "AND(user_installed_app=netflix, impression_app=pandora") as inputs. Good at memorization.
- Deep Model alone has a hard time to learning embedding for cross-product features because no data for most combinations but still makes predictions.
- Trained jointly on 500B examples.
