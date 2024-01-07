from torch.utils.data import Dataset


def pad_sequences(sequences, maxlen=200, padding_value=0):
    # Pad and/or truncate the sequences
    return [list(seq) + [padding_value] * (maxlen - len(seq)) if len(seq) < maxlen else seq[:maxlen] for seq in sequences]


def encode_sequences(sequences, gammas):
    encodings = {
        "A": 1,
        "U": 2,
        "G": 3,
        "C": 4,
        # We don't want to have these in the sequences, cause it's RNA
        "P": -10000,
        "T": -10000,
    }

    encoded_sequences = [[encodings[x.upper()] for x in seq]
                         for seq in sequences]
    encoded_gammas = gammas[:]
    i = 0
    while i < len(encoded_sequences):
        # Remove sequences with negative values
        if sum(encoded_sequences[i]) < 0:
            encoded_sequences.pop(i)
            encoded_gammas.pop(i)
        i += 1

    return encoded_sequences, encoded_gammas


def create_mask(sequences, padding_value=-1):
    return [[float(token != padding_value) for token in seq] for seq in sequences]
# Custom Dataset class


class NucleotideDataset(Dataset):
    def __init__(self, _sequences, _angles, _masks):
        self._sequences = _sequences
        self._angles = _angles
        self._masks = _masks

    def __getitem__(self, idx):
        return self._sequences[idx], self._angles[idx], self._masks[idx]

    def __len__(self):
        return len(self._sequences)
