import numpy as np
import time
import dnaEncoding as de


def chaotic_map(r, x):
    theta = (-r * x) + x ** 3 - (r * np.tan(x))
    newx = np.abs(np.abs(np.sin(theta)) - np.abs(np.cos(theta)))
    if np.isnan(newx):
        print(r, newx)
        exit(1)
    return newx


def derivative(r, x):
    theta_der = -r + 3 * (x ** 2) - r / (np.cos(x) ** 2)
    s_term = np.sin(-r * x + x ** 3 - r * np.tan(x))
    c_term = np.cos(-r * x + x ** 3 - r * np.tan(x))
    der = theta_der * ((np.abs(s_term) - np.abs(c_term)) * ((s_term * c_term) / (np.abs(c_term)) + (s_term * c_term) \
                                                            / (np.abs(s_term)))) / (
              np.abs(np.abs(s_term) - np.abs(c_term)))
    return der


def bifurcation(r_values, x_ini, transientIterations, stableIterations):
    if isinstance(r_values, int) or isinstance(r_values, float):
        r_values = [r_values]
    bifur_x = []
    bifur_r = []
    for r in r_values:
        x = x_ini

        for _ in range(transientIterations):
            x = chaotic_map(r, x)

        for i in range(stableIterations):
            x = chaotic_map(r, x)
            bifur_x.append(x)
            bifur_r.append(r)

    return bifur_r, bifur_x


def createSequence(controlParameter, initialValue, length):
    indices = np.arange(length)
    result = np.empty(length, dtype=np.float64)
    result[0] = chaotic_map(controlParameter, initialValue)
    for i in range(1,length):
        result[i] = chaotic_map(controlParameter, result[i-1])
    return result


def generateDnaString(lengthOfString, controlParameter, key):
    if key < 0.0 or key >= 1.0:
        print("Invalid key. Must be in range 0.0 to 1.0")
        return

    key = np.float64(key)  # Ensure key is a float
    sequence_length = determineSequenceLength(lengthOfString)

    # Generate chaotic map values for the determined sequence length
    numbers = createSequence(controlParameter, key, sequence_length)

    # Map numbers to DNA characters
    rule_used = np.array(['A', 'G', 'C', 'T'])
    dna_characters = rule_used[(numbers * 256).astype(np.uint8) & 3]

    # Extend the DNA sequence to match or exceed lengthOfString
    if len(dna_characters) < lengthOfString:
        dna_string = ''.join(dna_characters)
        num_repeats = lengthOfString // len(dna_string) + 1
        dna_string = (dna_string * num_repeats)[:lengthOfString]
    else:
        dna_string = ''.join(dna_characters[:lengthOfString])

    return dna_string


def newgenerateDnaString(lengthOfString, controlParameter, key):
    if key < 0.0 or key >= 1.0:
        print("Invalid key. Must be in range 0.0 to 1.0")
        return

    key = np.float64(key)  # Ensure key is a float
    sequence_length = determineSequenceLength(lengthOfString)

    # Generate chaotic map values for the determined sequence length
    numbers = createSequence(controlParameter, key, sequence_length)

    # Map numbers to DNA characters
    rule_used = np.array(['A', 'G', 'C', 'T'])
    dna_characters = rule_used[(numbers * 256).astype(np.uint8) & 3]

    # Extend the DNA sequence to match or exceed lengthOfString
    if len(dna_characters) < lengthOfString:
        dna_string = ''.join(dna_characters)
        num_repeats = lengthOfString // len(dna_string) + 1
        dna_string = (dna_string * num_repeats)[:lengthOfString]
    else:
        dna_string = ''.join(dna_characters[:lengthOfString])

    return dna_string


def determineSequenceLength(lengthOfString):
    if lengthOfString < 2000:
        return lengthOfString
    elif lengthOfString < 100000:
        return int(lengthOfString / 5)
    elif lengthOfString < 1000000:
        return int(lengthOfString / 10)
    else:
        return 1000000


if __name__ == "__main__":
    n = chaotic_map(3248.3456, 0.678)
    print(n)
    start = time.time()
    # dnaString = createDnaString(71912448, 8.9304854, 0.678)
    dna_sequence = generateDnaString(71912448,4948.9304854, 0.678)
    print(time.time() - start)
    # print(dna_sequence, len(dna_sequence))
