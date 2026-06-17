# Models

This directory contains the models used in the molecule-signature project. More details are available in the paper and the associated GitHub repositories.

## Weights

Model weights are provided as PyTorch/Lightning checkpoint files:

- `weights/pretrained.ckpt`: The weights of the pre-trained model trained on the eMolecules dataset.
- `weights/finetuned-fold0.ckpt`: The weights of the fine-tuned model trained on the MetaNetX dataset.

## Tokens

Tokens are provided as SentencePiece models. Both models were trained using the `sentencepiece` Python package on the eMolecules dataset.

- `tokens/ECFP.model`: The SentencePiece model for tokenizing ECFP fingerprints.
- `tokens/SMILES.model`: The SentencePiece model for tokenizing SMILES strings.

**Citation**: P Meyer, T Duigou, G Gricourt, JL Faulon, in preparation.
