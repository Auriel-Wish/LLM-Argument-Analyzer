<h1>This repository is for the trial project for the Tufts summer research internship.</h1>

<h2>How to run:</h2> The web application is not hosted online because that costs money. To run the web application locally, download the files and run "python3 Parts-2-and-3/main.py". Open the local link that the terminal says it is hosting the web app on, and enter an argument to be analyzed.

<h2>Part-1</h2> contains the code to train a model on an IBM dataset. It also contains scripts that test some argument inputs into both the IBM-trained model and a different trained model from HuggingFace.

<h2>Parts-2-and-3</h2> contains the code that runs the web app. The web app allows the user to input an argument and recieve analysis on it, including the strength of the argument and components of the argument (such as conflict, rephrase, ...etc). It allows for lengthy argument inputs. If the argument is too long to run through the analysis models, it is run through a summarization model that shortens the argument (but keeps its ideas). It does this over and over until the argument is short enough to pass through the analysis models. The user can choose whether to send the argument through a model trained on the IBM dataset or a different trained model from HuggingFace.
