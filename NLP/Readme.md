## Model objective
The model gets a text prompt describing a pokemon and has to predict what pokemon matches that description the most

## Data
Our training data is a mix of data pulled from multiple websites:

https://pokemon.fandom.com/wiki/

[https://www.kaggle.com/datasets/lantian773030/pokemonclassification](https://pokemon.fandom.com/wiki/)

https://pokemondb.net/pokedex/

https://bulbapedia.bulbagarden.net/wiki

The following was used to match metadata with the pokemon from the other data we scraped:

https://www.kaggle.com/datasets/rounakbanik/pokemon

## Setup info
| type | value |
| --- | --- |
| total sentences | ~10.800 sentences for 151 pokemon (first gen only) |
| training duration | heavily depends on available hardware and model size. training runs took anything from 1.5 hours to 18 hours for the large bert |

## folders
*everything relevant can be found in the assignment folder*

| folders | content |
| ------------------ | ----------------- |
| [*data*](https://github.com/skramer-dev/ai-lab/tree/main/NLP/assignment/data)  | everything around the data that was used for training and inference |
| [*finetuning*](https://github.com/skramer-dev/ai-lab/tree/main/NLP/assignment/finetuning/) | multiple notebooks for the finuting approach, fully trained models were not saved to git due to their large size |
| [*prompting*](https://github.com/skramer-dev/ai-lab/tree/main/NLP/assignment/prompting/) | multiple notebooks for the prompting approach, fully trained models were not saved to git due to their large size |

## Notebooks

| Notebook | content |
| ------------------ | ----------------- |
| [*inference_comp*](assignment/inference_comp.ipynb) | a lot of code just to have a fancy comparison between the predictions of the different models |
| [*finetuning/bert-base*](assignment/finetuning/bert-base.ipynb) | fintuning notebook for a bert-base model, this one was trained locally on weaker hardware |
| [*finetuning/bert-large-remote*](assignment/finetuning/bert-large-remote.py) | we converted the bert-base notebook into a normal python file to be able to run it remotely on better hardware. Functionally the same as the regular bert notebook except using a bert large |
| [*finetuning/Inference*](assignment/finetuning/inference.ipynb) | first iteration of inference code just for the bert model we trained initially. For better comparison use the 'inference_comp' notebook |
| [*prompting/bert-local*](assignment/prompting/bert-local.ipynb) | prompting notebook for a bert-base model, once again trained locally |
| [*prompting/bert-large-local*](assignment/prompting/bert-large-local.ipynb) | failed attempt at training a large bert locally, results didn't look good |
| [*prompting/gpt2-local*](assignment/prompting/gpt2-local.ipynb) | prompting notebook for a gpt2 model, trained locally and with pretty good results |
| [*prompting/gpt2-large-remote*](assignment/prompting/gpt-2-large-remote.py) | converted notebook for training a large gpt2 remote (this one ultimately didn't work out because the remote machines didn't have the package versions that were needed and we didn't have root to fix it |
| [*prompting/inference_prompting*](assignment/prompting/inference_prompting.ipynb) | inference notebook just for the prompting models, better comparison once again in the 'inference_comp' notebook |
