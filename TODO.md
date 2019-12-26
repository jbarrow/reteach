

- [ ] incorporate multiple levels of feature embeddings
  - [ ] token-level (e.g. PoS, Case, etc.)
  - [x] exercise-level
- [ ] incorporate regression features
  - [x] include them in the model (AUC: 82.0)
  - [ ] ensure that they're 0/1 normed
  - [ ] impute reasonable values for missing data
- [ ] incorporate more advanced features about when the last time a user
      saw a particular term (or perhaps disambiguated term), perhaps with
      some decay model
- [x] write a predictor and evaluate some of the things the model gets wrong
- [ ] ensure the reader/model allows me to add/remove features quickly for experimentation
- [ ] incorporate prior knowledge about term difficulty, perhaps in the form
      of a CEFR-level embedding for each term.
- [ ] create a user model to track user progress through time
- [x] incorporate F1 metric
- [ ] run an allentune job to find good hyperparameters

Biggest gains will probably be from:

  1. correctly handling the different feature embeddings
  2. incorporating regression features
  3. incorporating CEFR levels per token
  4. incorporating prior time of having seen the term/concept
  5. the user model

Thought: what if I train a model that attempts to move a user embedding
         closer to the words they don't know at any given time. Or perhaps
