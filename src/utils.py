from torch.utils.data import Dataset
from math import isnan


def encode_sequences(sequences, gammas):
    encodings = {
        "A": 0,
        "U": 1,
        "G": 2,
        "C": 3,
    }

    encoded_sequences = [[encodings[x.upper()] for x in seq]
                         for seq in sequences]
    encoded_gammas = gammas[:]
    return encoded_sequences, encoded_gammas


def decode_sequences(sequences):
    decodings = {
        0: "A",
        1: "U",
        2: "G",
        3: "C",
    }

    decoded_sequences = sequences.tolist()

    decoded_sequences = [[decodings[x] for x in seq]
                         for seq in decoded_sequences]

    return ["".join(x) for x in decoded_sequences]


def create_mask(sequences, padding_value=0):
    return [[float(token != padding_value) for token in seq] for seq in sequences]
# Custom Dataset class


def process_data(sequences, gammas):
    encoded_sequences, encoded_gammas = encode_sequences(sequences, gammas)
    # if any gamma is nan remove sequence, gamma and mask
    i = 0
    while i < len(encoded_gammas):
        for j in range(len(encoded_gammas[i])):
            if isnan(encoded_gammas[i][j]):
                encoded_sequences.pop(i)
                encoded_gammas.pop(i)

        i += 1
    return encoded_sequences, encoded_gammas


def calculate_class_index(angle, num_classes):
    # Calculate the class index
    if angle < 0:
        angle += 360
    class_index = int(angle // (360/num_classes))
    # If the angle is exactly 180, we need to subtract 1 from the class index
    if angle == 180:
        class_index -= 1
    return class_index


def convert_classes_to_angles(classes, num_classes):
    # Convert the classes the average angle in range
    return [class_index * 360/num_classes + 360/num_classes/2 for class_index in classes]


class NucleotideDataset(Dataset):
    def __init__(self, _sequences, _angles):
        self._sequences = _sequences
        self._angles = _angles

    def __getitem__(self, idx):
        return self._sequences[idx], self._angles[idx]

    def __len__(self):
        return len(self._sequences)
