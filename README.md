# job_recommender_prototpye
Generate job recommendation based on your CV (pdf file)



## Description

* This project is using a recommendation system based on LDA and TF-IDF scoreCancel changes
* The baseline of the model is based on this project: https://github.com/ethHong/-Course-Recommentation-Project
* When Resume / CV is uploaded as pdf file, the model
  * Extract text
  * Cleanse, filter stopwords and create LDA topic model
  * Compute JSD of topic distribution of input (resume) with each job desctriptions, with TF-IDF keyword score
  * Generate top 10 relevant job descriptions

* Formula to compute distances between topic distribution of Resume and JD is:

$$[Score_{tfidf}(C) = \sum_{i-1}^k tfidf(C, w_i)]$$

$$[Score_{LDA-dist}(C) = \frac{1}{D_{js}(D, C)_{scaled} + 0.01}]$$



## Example

* Input page
* Example Resume input
* Example output
