<h1>This repository is for the trial project for the Tufts summer research internship.</h1>

<h2>How to run:</h2> The web application is not hosted online because that costs money. To run the web application locally, download the files, create a virtual environment, pip install the requirements, and run "python3 Parts-2-and-3/main.py". Open the local link that the terminal says it is hosting the web app on, and enter an argument to be analyzed.

<h2>Part-1</h2> This folder contains the code to train a model on an IBM dataset (train.py). It also contains scripts that test some argument inputs into both the IBM-trained model and a different trained model from HuggingFace (main[...].py). The IBM-trained model can be found at <a href="https://huggingface.co/aurielwish/trial-project">https://huggingface.co/aurielwish/trial-project</a>. Although the IBM dataset contains 30000 arguments, the model was only trained on 3000 because of the limitations of my computer - it took 6 hours to train it on this subset and it would freeze if I tried to do anything else on my computer.

<h2>Parts-2-and-3</h2> This folder contains the code that runs the web app. The backend code is in funcs.py and main.py. The frontend code is in templates/index.html. The web app allows the user to input an argument and recieve analysis on it. It allows for lengthy argument inputs. If the argument is too long to run through the analysis models, it is run through a summarization model that shortens the argument (but keeps its ideas). It does this over and over until the argument is short enough to pass through the analysis models. The user can choose whether to send the argument through a model trained on the IBM dataset or a different trained model from HuggingFace.

The analysis performed on the argument includes:
<ol>
    <li>Strength of the argument (number between 0 and 1)</li>
    <li>Components of the argument (such as conflict, rephrase, ...etc)</li>
    <li>Chatbot analysis of the argument that comments on why the argument is/isn't good</li>
</ol>
The first 2 tasks occur together and do not take much time. The third task occurs after and takes more time.