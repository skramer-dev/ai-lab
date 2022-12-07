## Model objective
The model gets a text prompt describing a pokemon and has to predict what pokemon matches that description the most

## Data
Our training data is a mix of data pulled from multiple websites:

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
| [*data*](/assignment/data)  | everything around the data that was used for training and inference |
| [*finetuning*](/assignment/finetuning) | multiple notebooks for the finuting approach, fully trained models were not saved to git due to their large size |
| [*prompting*](/assignment/prompting) | multiple notebooks for the prompting approach, fully trained models were not saved to git due to their large size |

## Notebooks

| Notebook | content |
| ------------------ | ----------------- |
| [*inference_comp*] | a lot of code just to have a fancy comparison between the predictions of the different models |
| [*finetuning/bert-base*] | fintuning notebook for a bert-base model, this one was trained locally on weaker hardware |
| [*finetuning/bert-large-remote*] | we converted the bert-base notebook into a normal python file to be able to run it remotely on better hardware. Functionally the same as the regular bert notebook except using a bert large |
| [*finetuning/Inference*] | first iteration of inference code just for the bert model we trained initially. For better comparison use the 'inference_comp' notebook |
| [*prompting/bert-local*] | prompting notebook for a bert-base model, once again trained locally |
| [*prompting/bert-large-local*] | failed attempt at training a large bert locally, results didn't look good |
| [*prompting/gpt2-local*] | prompting notebook for a gpt2 model, trained locally and with pretty good results |
| [*prompting/gpt2-large-remote*] | converted notebook for training a large gpt2 remote (this one ultimately didn't work out because the remote machines didn't have the package versions that were needed and we didn't have root to fix it |
| [*prompting/inference_prompting*] | inference notebook just for the prompting models, better comparison once again in the 'inference_comp' notebook |
