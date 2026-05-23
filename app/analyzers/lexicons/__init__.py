"""
Lexicons & scoring augmentation for Spanish sentiment analysis.

Why this layer exists
---------------------
The base Spanish sentiment model (`pysentimiento/robertuito-sentiment-analysis`)
is trained on tweets, which is a different register from intimate WhatsApp
conversations: it under-rates affection ("amor", "te amo", diminutives, ❤️) and
mis-reads hyperbolic playfulness ("me muero", "convulsiono", "WTF amor") as
negative.

This package supplies a *late-fusion* layer that combines:

  1. The base ML score from pysentimiento (kept as the strongest signal).
  2. A Spanish lexicon score (NRC + AFINN Spanish translations, 7.6k words).
  3. An intimate-couple-vocabulary score (hand-curated, ~250 entries for
     vocatives, diminutives, hyperbolic affection, playful sarcasm).
  4. An emoji score (Kralj Novak 2015 Emoji Sentiment Ranking, 970 emojis).

When a layer has no signal for a message, it abstains; the remaining layers
keep their weights proportional. This is the standard hybrid-lexicon approach
documented in Aragón et al. (2020).

Citations & licensing — see CITATIONS.md.
"""
