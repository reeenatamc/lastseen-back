# Sentiment lexicon citations & licenses

The hybrid Spanish sentiment scorer in this package combines four data
sources. Attribution for each is below; please keep this file in any
redistribution of the package.

---

## 1. Emoji Sentiment Ranking 1.0

File: `emoji_sentiment.csv`

> Kralj Novak P, Smailović J, Sluban B, Mozetič I. *Sentiment of Emojis*.
> PLOS ONE 10(12): e0144296 (2015).
> <https://doi.org/10.1371/journal.pone.0144296>

License: **Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)**.
Source: <http://hdl.handle.net/11356/1048> (CLARIN.SI).

Commercial use permitted with attribution under the same license.

---

## 2. NRC Word-Emotion Association Lexicon (Spanish translation)

The Spanish words in `lexicon_es.csv` derived from the NRC lexicon trace
back to:

> Mohammad SM, Turney PD. *Crowdsourcing a Word-Emotion Association
> Lexicon.* Computational Intelligence 29(3): 436-465 (2013).
> <https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm>

License: **NRC Research-Only / non-commercial.** Commercial use requires
contacting Saif M. Mohammad. See the official terms at the URL above.

---

## 3. AFINN (Spanish adaptation)

The valence scores for Spanish words in `lexicon_es.csv` derived from AFINN
trace back to:

> Nielsen FÅ. *A new ANEW: Evaluation of a word list for sentiment analysis
> in microblogs.* In Proceedings of the ESWC Workshop on 'Making Sense of
> Microposts' (2011).
> <https://arxiv.org/abs/1103.2903>

Spanish translation mirror referenced from
<https://github.com/jboscomendoza/lexicos-nrc-afinn>.

License: AFINN is released under the Open Database License v1.0 (ODbL).

---

## 4. Couple-chat intimate-Spanish vocabulary (`intimate_es.py`)

Hand-curated by the LastSeen project for the specific register of intimate
WhatsApp conversations in Latin-American Spanish. Original work, MIT-licensed
together with the rest of this repository.

---

## Background reading for the hybrid-lexicon approach

The late-fusion design implemented in `scorer.py:hybrid_score` follows the
standard hybrid lexicon + ML pattern documented in:

- Aragón ME, Monroy AP, Montes-y-Gómez M, López-Monroy AP, Escalante HJ.
  *Improved emotion recognition in Spanish social media through the
  incorporation of lexical knowledge.* Future Generation Computer Systems
  110 (2020). <https://doi.org/10.1016/j.future.2019.09.034>

For the emphasis multiplier:

- Brody S, Diakopoulos N. *Cooooooooooooooollllllllllllll!!!!!!!!!!!!!!*
  EMNLP (2011) — established that orthographic stretching (`amoor`,
  `nooooo`) is a quantifiable affect-intensifier.
