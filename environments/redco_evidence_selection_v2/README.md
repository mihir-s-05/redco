# ReDCO deterministic evidence selection v2

This taskset defines the narrowed Stage D objective:

- one QASPER paper is written to the RLM workspace;
- the policy returns a Python list of nonempty verbatim spans;
- every predicted span must occur exactly in the paper;
- character-union precision, recall, and F1 are deterministic;
- invalid, empty, hallucinated, or non-list output receives reward zero.

It is not an alphaXiv/SkyRL reward reproduction.
