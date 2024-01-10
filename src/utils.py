from torch.utils.data import Dataset
from math import isnan


def pad_sequences(sequences, gammas, maxlen=200, padding_value=0):
    # We put poly-A tail in the end, but it doesn't matter since it's masked :P
    # Pad and/or truncate the sequences
    padded_seqs = [list(seq) + [padding_value] * (maxlen - len(seq))
                   if len(seq) < maxlen else seq[:maxlen] for seq in sequences]
    padded_gammas = [list(gamma) + [padding_value] * (maxlen - len(gamma))
                     if len(gamma) < maxlen else gamma[:maxlen] for gamma in gammas]
    masks = [[1] * len(seq) + [0] * (maxlen - len(seq))
             if len(seq) < maxlen else [1] * maxlen for seq in sequences]

    return padded_seqs, padded_gammas, masks


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


def decode_sequences(sequences, masks=None):
    decodings = {
        0: "A",
        1: "U",
        2: "G",
        3: "C",
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


def read_fasta(in_path: str):
    sequences = list()
    names = list()
    current_sequence = ""

    with open(in_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('>'):
                # If a new sequence header is found, append the previous sequence (if any) to the list
                if current_sequence:
                    sequences.append(current_sequence)
                # Start a new sequence
                current_sequence = ""
                names.append(line[1:])
            else:
                # Append the current line to the current sequence
                current_sequence += line

        # Append the last sequence in the file
        if current_sequence:
            sequences.append(current_sequence)
    return sequences, names


def fasta_preprocess(sequences_list: [str]):
    encoded_sequences, _ = encode_sequences(sequences_list, [])
    padded_sequences, _, masks = pad_sequences(
        encoded_sequences, encoded_sequences)
    return padded_sequences, masks


def process_data(sequences, gammas, maxlen=200):
    encoded_sequences, encoded_gammas = encode_sequences(sequences, gammas)
    padded_sequences, padded_gammas, masks = pad_sequences(
        encoded_sequences, encoded_gammas, maxlen=maxlen)

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
