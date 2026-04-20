# tfidf-hadoop-java-pig
TF-IDF implementation using Hadoop MapReduce (Java) and Pig for term weighting and top-k keyword extraction on a small text corpus.

This repository contains two small text-processing implementations built around term weighting on a document collection:

- a **Java Hadoop MapReduce** version
- a **Pig** script version

The goal of the project is to process a small set of text documents, compute term-based scores, and return the most important terms for each document.

## What's included

- `TFIDF.java`  
  A Java implementation built as a multi-stage Hadoop MapReduce workflow.

- `tfidf.pig`  
  A Pig script that follows a similar document-processing pipeline using Pig relations and transformations.

## What the project does

Given a small corpus of text files, the project:

- loads and tokenizes the text
- counts term occurrences per document
- computes document-level statistics
- calculates term weights
- ranks terms by score
- keeps the top results for each document

In practice, the output is a list of the highest-scoring terms for every input document.

## Java implementation

The Java version is organized as a sequence of MapReduce jobs:

1. **Count terms and document length**  
   Counts how many times each term appears in each document and also tracks total document length.

2. **Compute TF values**  
   Uses the term counts and document length to calculate term frequency for each document.

3. **Compute corpus-level weighting**  
   Groups values by term and computes an IDF-style score.

4. **Rank terms per document**  
   Multiplies the intermediate values and keeps the top 5 terms for each document.

This version was written as a small Hadoop MapReduce exercise and is currently set up for local execution.

## Pig implementation

The Pig script follows the same general idea in a more compact, data-flow style:

- load input documents
- assign document IDs
- tokenize text
- group and count terms
- compute per-document values
- join intermediate relations
- calculate final scores
- keep the top-ranked terms per document

## Technologies used

- Java
- Hadoop MapReduce
- Apache Pig
- Basic text processing / document scoring concepts

## Input format

The scripts assume a small set of text documents as input.

Example structure:

```bash id="vyn6by"
input/
  input1.txt
  input2.txt
  input3.txt
