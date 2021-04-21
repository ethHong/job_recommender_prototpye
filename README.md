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

![Presentation1](https://user-images.githubusercontent.com/43837843/115522825-6a448880-a2c7-11eb-8347-c26550a27b73.png)

* Formula to compute distances between topic distribution of Resume and JD is:
![gif latex](https://user-images.githubusercontent.com/43837843/115522794-61ec4d80-a2c7-11eb-8858-0ddf5400cdd1.gif)
![gif latex-2](https://user-images.githubusercontent.com/43837843/115522741-57ca4f00-a2c7-11eb-92d3-6811e20d083e.gif)


## Example

* Input page
<img width="1424" alt="Screen Shot 2021-04-21 at 5 10 31 PM" src="https://user-images.githubusercontent.com/43837843/115522898-7af4fe80-a2c7-11eb-9d20-d35014830f48.png">
* Example Resume input
![Screen Shot 2021-04-21 at 5 32 51 PM](https://user-images.githubusercontent.com/43837843/115523033-9a8c2700-a2c7-11eb-8c25-e7b82933a09d.png)
* Example output
<img width="1336" alt="Screen Shot 2021-04-21 at 5 10 55 PM" src="https://user-images.githubusercontent.com/43837843/115523069-a37cf880-a2c7-11eb-8322-db32cfc18bfe.png">
