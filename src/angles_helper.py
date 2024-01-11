import argparse
import os
import torch
import json

from utils import read_fasta, fasta_preprocess

model2path = {
    'binary': 'BinClassifier.pt',
    # 'tertiary': ,
    '20': 'TwentyClassifier.pt',
    '30': 'ThirtyClassifier.pt',
    'regression': 'Regressor.pt'
}

model2classes = {
    'binary': 2,
    # 'tertiary': ,
    '20': 20,
    '30': 30,
    'regression': None
}


class AngleHelper:
    def __init__(self, in_path: str, out_path: str, model_type: str):
        self.in_path = in_path
        self.out_path = out_path
        self.model_type = model_type

    def predict(self, in_path: str, out_path: str, model_type: str):
        """
        Function that predicts the angles for the sequence of nucleotides
        Args:
        - in_path: path to a `fasta` file.
            Example:
            "
            >17EP
            ACGUUCU
            "
        - out_path: path to a `.json` file where will be stored the prediciton.
            It should have the following keys:
            {
                "17EP": {
                "sequence": "ACGUUCU",
                "angles": {"gamma": [0, 0, 0, 0, 0, 0, 0]}
                }
            }
        - model_type: the model to be used for inference, the choice is between [`binary`, `tertiary`, `20`, `30`, `regression`]'
        """

        print(f"Predicting angles for {in_path} using {model_type} model")
        sequences_list, names_list = read_fasta(in_path)
        padded_sequences, masks = fasta_preprocess(sequences_list)

        model_path = os.path.join("../models", model2path[model_type])
        model = torch.load(model_path)
        model.eval()

        outputs = model(torch.tensor(padded_sequences, dtype=torch.long))

        num_classes = model2classes[model_type]
        if num_classes is not None:
            outputs = torch.argmax(outputs, dim=2)

        tmasks = torch.tensor(masks).bool()
        outputs_masked = [out[tmask] for out, tmask in zip(outputs, tmasks)]

        if num_classes is not None:
            outputs_masked = list(map(lambda class_index: class_index *
                                      360/num_classes + 360/num_classes/2, outputs_masked))

        result_json = {}

        for i in range(len(outputs_masked)):
            result_json[names_list[i]] = {
                "sequence": sequences_list[i],
                "angles": {"gamma": outputs_masked[i].tolist()}
            }

        with open(out_path, "w") as f:
            json.dump(result_json, f)
        return None


if __name__ == "__main__":
    in_path = os.path.join("data", "sample", "example.fasta")
    out_path = "sample.json"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_path',
        action='store',
        required=True,
        type=str,
        help='absolute path to the input fasta file'
    )

    parser.add_argument(
        '--out_path',
        action='store',
        required=True,
        type=str,
        help='absolute path to the output .json file'
    )

    parser.add_argument(
        '--model_type',
        action='store',
        required=True,
        type=str,
        help='model type to use for inference, the choice is between [`binary`, `tertiary`, `20`, `30`, `regression`]'
    )

    argv = vars(parser.parse_args())
    args = parser.parse_args()

    for key, value in argv.items():
        print(f'{key} is {value}')

    angle_helper = AngleHelper(args.input_path, args.out_path, args.model_type)
    angle_helper.predict(args.input_path, args.out_path, args.model_type)
