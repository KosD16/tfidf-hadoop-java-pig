# tfidf-hadoop-java-pig
TF-IDF implementation using Hadoop MapReduce (Java) and Pig for term weighting and top-k keyword extraction on a small text corpus.
# TF-IDF with Hadoop MapReduce and Pig

This repository contains two implementations of TF-IDF computation over a small text corpus:

- a **Java Hadoop MapReduce** pipeline
- a **Pig Latin** script-based workflow

The project computes **Term Frequency (TF)**, **Inverse Document Frequency (IDF)**, and the **top 5 highest-scoring TF-IDF terms per document**.

## Technologies
- Java
- Hadoop MapReduce
- Apache Pig
- Text processing / Information Retrieval concepts

## Project Structure
- `TFIDF.java` – Java implementation using a 4-job Hadoop MapReduce pipeline
- `tfidf.pig` – Pig Latin implementation of the same TF-IDF workflow

## What the project does
Given a small collection of text documents, the project:

1. tokenizes the text
2. computes term frequency for each document
3. computes inverse document frequency across the corpus
4. calculates TF-IDF scores
5. returns the **top 5 terms** for each document based on TF-IDF

## Java MapReduce Workflow
The Java implementation is split into 4 stages:

### Job 1 – Term counts and document length
Counts:
- occurrences of each term in each document
- total number of terms per document

### Job 2 – TF computation
Reorganizes intermediate results by document and computes:

TF(term, doc) = term_count / document_length

### Job 3 – IDF computation
Groups values by term and computes:

IDF(term) = log(N / df)

where:
- `N` = total number of documents
- `df` = number of documents containing the term

### Job 4 – TF-IDF ranking
Computes:

TF-IDF = TF × IDF

Then sorts terms by score and keeps the **top 5** for each document.

## Pig Workflow
The Pig implementation follows a similar approach:

- loads the input documents
- assigns document IDs
- tokenizes text
- computes term counts
- computes document lengths
- calculates TF
- calculates IDF
- joins TF and IDF
- computes TF-IDF
- keeps the top 5 terms per document

## Input
The implementation is designed for a small corpus of text files.

Example input structure:
```bash
input/
  input1.txt
  input2.txt
  input3.txt
