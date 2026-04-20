
doc1 = LOAD '$input/input1.txt' USING TextLoader() AS (text:chararray);
doc2 = LOAD '$input/input2.txt' USING TextLoader() AS (text:chararray);
doc3 = LOAD '$input/input3.txt' USING TextLoader() AS (text:chararray);


d1 = FOREACH doc1 GENERATE 'input1.txt' AS doc_id, text;
d2 = FOREACH doc2 GENERATE 'input2.txt' AS doc_id, text;
d3 = FOREACH doc3 GENERATE 'input3.txt' AS doc_id, text;

raw_data = UNION d1, d2, d3;

tokenized = FOREACH raw_data GENERATE doc_id, FLATTEN(TOKENIZE(text)) AS term;


grp_term_doc = GROUP tokenized BY (doc_id, term);
term_counts = FOREACH grp_term_doc GENERATE 
    group.doc_id AS doc_id, 
    group.term   AS term, 
    COUNT(tokenized) AS raw_count;

grp_doc = GROUP tokenized BY doc_id;
doc_lengths = FOREACH grp_doc GENERATE 
    group AS doc_id, 
    COUNT(tokenized) AS total_terms;

joined_tf = JOIN term_counts BY doc_id, doc_lengths BY doc_id;


tf_data = FOREACH joined_tf GENERATE 
    term_counts::doc_id AS doc_id,
    term_counts::term   AS term,
    ((double)term_counts::raw_count / (double)doc_lengths::total_terms) AS tf_val;

grp_term_corpus = GROUP tf_data BY term;

idf_data = FOREACH grp_term_corpus GENERATE 
    group AS term,
    LOG(3.0 / SUM(tf_data.tf_val)) AS idf_val;


joined_tfidf = JOIN tf_data BY term, idf_data BY term;

final_scores = FOREACH joined_tfidf GENERATE 
    tf_data::doc_id AS doc_id,
    tf_data::term   AS term,
    (tf_data::tf_val * idf_data::idf_val) AS tfidf_score;


grp_final = GROUP final_scores BY doc_id;

top_5_results = FOREACH grp_final {
    sorted = ORDER final_scores BY tfidf_score DESC;
    top5   = LIMIT sorted 5;
    GENERATE FLATTEN(top5);
};

formatted_output = FOREACH top_5_results GENERATE 
    doc_id, 
    CONCAT(term, CONCAT(' -> ', (chararray)tfidf_score));

STORE formatted_output INTO '$output';
