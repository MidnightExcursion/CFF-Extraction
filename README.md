# CFF-Extraction
An ANN-approach to extracting what's called the "Compton Form Factors" that are important in the "Proton Spin Crisis."

# A Physics Introduction

What's called the "Proton Spin Crisis" is a major problem in modern physics. The idea is roughly as follows:

The Proton is not a fundamental particle. Rather, it is composed of fundamental consitutents that interact in complex ways. Specifically, we know that its constituents are Quarks and Gluons. These are fundamental particles as described in the Quantum Field Theory called "Quantum Chromodynamics." In essence, the fundamental particles have properties -- like quantum spin -- that when they come together to form a composite object, we might expect those properties to "add" in a simple way.

The Proton, as we understand it, has a half-integer spin (1/2). The observation that the Proton's spin seems to not be the sum of its parts is the genesis of the tension. Those Quarks and Gluons contribute their spins to the overall net spin of the Proton in an unintuitive way. As a pop-science example, it's known that the Proton contains three valence Quarks: Two Ups and one Down Quark. Each of those three Quarks actually has half-integer spin. Naively, then, if spin "adds" (it doesn't in the way that 1 + 1 = 2 "adds"), then the spin of the Proton ought to be 3/2. It is not.

# The Objective of the Code

Use a DNN approach to try to find the functional form of the Compton Form Factors (CFFs).

# The "Chronology" of the Code:

1. Verify there Exists Data:

    1.1. We first check to see if there provided argument for `-d` (filepath) actually has something at that filepath.

    1.2. Then, we see if we can read that data with Pandas' `read_csv()` function.

    1.3. Attempt to partition the entire dataframe into a smaller one based on the parameter for the `kin` argument, which is just an integer that refers to the kinematic set of the data, and is represented in the data with the column `['set']`.

2. Begin the Replica Method:

    2.1. Verify intact Directory Structure:

        2.1.1. 

# The Replica Method Summarized:

## The "Chronology" of the Procedure:

1. We specify the number of Replicas we want to run. That is a number that we will symbolically call $N_{\text{replicas}}$. 

2. We then generate "pseudodata" for each Replica. That will become the training, testing, and validation data for our ANN.

3. We split the pseudodata into the training, testing, and validation data.

4. Then, we initialize the DNN. It's critical to understand the that DNN is exactly the same every time apart from the random sampling of the weights and biases upon initialization of the DNN.

5. We train the DNN for $N_{\text{epochs}}$. 

6. Then, we see what the results are for that Replica. These come in the form of fancy plots.

7. That was all for a single Replica. 
