from torch.utils.data import Dataset
from math import isnan


def pad_sequences(sequences, maxlen=200, padding_value=0):
    # Pad and/or truncate the sequences
    return [list(seq) + [padding_value] * (maxlen - len(seq)) if len(seq) < maxlen else seq[:maxlen] for seq in sequences]


def encode_sequences(sequences, gammas):
    encodings = {
        "A": 1,
        "U": 2,
        "G": 3,
        "C": 4,
    }

    encoded_sequences = [[encodings[x.upper()] for x in seq]
                         for seq in sequences]
    encoded_gammas = gammas[:]
    return encoded_sequences, encoded_gammas


def decode_sequences(sequences, masks=None):
    decodings = {
        1: "A",
        2: "U",
        3: "G",
        4: "C",
    }

    decoded_sequences = sequences.tolist()

    # Remove padding
    if masks is not None:
        for i in range(len(decoded_sequences)):
            decoded_sequences[i] = decoded_sequences[i][:int(sum(masks[i]))]

    decoded_sequences = [[decodings[x] for x in seq]
                         for seq in decoded_sequences]

    return ["".join(x) for x in decoded_sequences]


def create_mask(sequences, padding_value=0):
    return [[float(token != padding_value) for token in seq] for seq in sequences]
# Custom Dataset class


def process_data(sequences, gammas, maxlen=200):
    encoded_sequences, encoded_gammas = encode_sequences(sequences, gammas)
    padded_sequences = pad_sequences(encoded_sequences, maxlen=maxlen)
    padded_gammas = pad_sequences(encoded_gammas, maxlen=maxlen)
    masks = create_mask(padded_sequences)

    # if any gamma is nan remove sequence, gamma and mask
    i = 0
    while i < len(padded_gammas):
        for j in range(len(padded_gammas[i])):
            if isnan(padded_gammas[i][j]):
                padded_sequences.pop(i)
                padded_gammas.pop(i)
                masks.pop(i)
        i += 1
    return padded_sequences, padded_gammas, masks


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
    def __init__(self, _sequences, _angles, _masks):
        self._sequences = _sequences
        self._angles = _angles
        self._masks = _masks

    def __getitem__(self, idx):
        return self._sequences[idx], self._angles[idx], self._masks[idx]

    def __len__(self):
        return len(self._sequences)
