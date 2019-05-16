import os
import numpy as np

def loadGloveModel(gloveFile):
    """
    Given a vector file it loads the word and the corresponding word vector
    into a dictionary and returns it. 
    To avoid the long times taken to load this, we cache the loaded vectors
    into a .npy file the first time this function is called.
    args:
        gloveFile (str): Name of the vector file. Each line should have 
            "word v1 v2 v2 ... vd" where v1, v2, v3 ... vd are the d dimensions
            of the word vector.
    returns:
        model (dict): a mapping of form {"word": word_vector}
    """
    print ("Loading Glove Model")

    if os.path.exists("{}.npy".format(gloveFile[:-4])):
        return np.load("{}.npy".format(gloveFile[:-4]))[()]
    
    with open(gloveFile, encoding="utf8" ) as f:
        model = {}
        for line in f:
            splitLine = line.split()
            word = splitLine[0]
            embedding = np.array([float(val) for val in splitLine[1:]])
            model[word] = embedding
    
    print ("Done.",len(model)," words loaded!")
    np.save("{}.npy".format(gloveFile[:-4]), model)

    return model

def get_tag_words(fname):
    """
    Given a data file of format (word tag) in each line, it returns a dictionary
    with key (word, tag) and value number of times the combination appears if the
    tag is not "O"
    args:
        fname (str): Name of the datafile. Each line of datafile should contain
            "word tag". An empty line indicates that a description has ended.
    returns:
        tag_words (dict): A dict of format {(word, tag): count}
    """
    print("getting tags and words from {}".format(fname))
    
    tag_words = {}
    with open(fname) as fr:
        for line in fr:
            if line.strip() and len(line.strip().split(" ")) == 2:
                word, tag = line.strip().split(" ")
                if tag != "O":
                    key = (word, tag)
                    tag_words[key] = tag_words.get(key, 0) + 1
    
    return tag_words

def calculate_similarity(v1, v1_tag_words, v2, v2_tag_words, vector_dict):
    """
    Given two dicitonaries, with keys as the (word, tag) pairs and values as
    the counts of number of times they appear in their dataset,
    for each key in dictionary 1, we calculate similarity with key in dictionary
    2. We return a dictionary with key str(word1, tag1, word2, tag2, score) and
    the value as the number of times the two pairs appear together. 
    args:
        v1 (str) : name of the vertical v1
        v1_tag_words (dict): a mapping which contains (word, tag) as key and count
            as values, for words and tags from dataset of v1 vertical.
        v2 (str) : name of the vertical v2
        v2_tag_words (dict): a mapping which contains (word, tag) as key and count
            as values, for words and tags from dataset of v2 vertical.
        vector_dict (dict): a mapping which contains word as key and value as the
            word vector.
    """
    print("Calculating similarity for {} and {}".format(v1, v2))
    v1_sim_results = {}
    
    for (v1_word, v1_tag), v1_count in v1_tag_words.items():
        
        if v1_word not in vector_dict:
            continue
        
        v1_word_vec = vector_dict[v1_word]
        norm_v1_word_vec = np.linalg.norm(v1_word_vec)
        
        for (v2_word, v2_tag), v2_count in v2_tag_words.items():
        
            if v2_word not in vector_dict:
                continue
        
            v2_word_vec = vector_dict[v2_word]
            norm_score = norm_v1_word_vec * np.linalg.norm(v2_word_vec)
            cosine_similarity = 0 if norm_score == 0 else np.dot(v1_word_vec, v2_word_vec)/norm_score

            if cosine_similarity > 0.5:
                v1_key = "{} {}_{} {} {}_{}\t{}\n".format(v1_word, v1_tag, v1, v2_word, v2_tag, v2, cosine_similarity)
                # if a pair (a1, b1) appears 5 times in dataset 1
                # and if a pair (a2, b2) appear 10 times in dataset 2
                # then they appear together 5 * 10 times.
                v1_sim_results[v1_key] = v1_count * v2_count

    return v1_sim_results

def calculate_lambda(similarity_dict, outfile):
    """
    Calculates lambda values using given similarity values. Lambda values are
    calculated for each tag pair. They are stored in outfile, where each line
    is of the format "v1_tag v2_tag\tcount_of_times_the_tags_appeared_together\tlambda"
    args:
        similarity_dict (dict): the keys of the dictionary are of the format
            "v1_word v1_tag v2_word v2_tag" and the value is the cosine similarity
            between v1_word and v2_word
        outfile (str): The file where lambda values are to be written.
    returns:
        None
    
    """
    print("Calculating lambda to be stored in {}".format(outfile))
    gt_8_count = {}
    gt_5_count = {}
    tot_count = {}

    for key, count in similarity_dict.items():
    
        tag_word_pair, word_sim = key.split("\t")
        _, tag1, _, tag2 = tag_word_pair.strip().split(" ")
        k = (tag1, tag2)
        tot_count[k] = tot_count.get(k, 0) + count
    
        if float(word_sim) >= 0.8:
            gt_8_count[k] = gt_8_count.get(k, 0) + count
    
        if float(word_sim) >= 0.5:
            gt_5_count[k] = gt_5_count.get(k, 0) + count
    
    with open(outfile, "w") as fw:
        for key in tot_count:
            total_count = tot_count[key]
            above_8_count = gt_8_count.get(key, 0)
            above_5_count = gt_5_count.get(key, 0)
            lamda = 0 if above_5_count == 0 else above_8_count/above_5_count
            fw.write("{} {}\t{}\t{}\n".format(key[0], key[1], total_count, lamda))

if __name__ == "__main__":
    vecs = loadGloveModel("glove.6B.300d.txt")
    v1 = "mangal"
    v2 = "necklace"
    v1_list = get_tag_words("mangal.txt")
    v2_list = get_tag_words("necklace.txt")
    sim_results = calculate_similarity(v1, v1_list, v2, v2_list, vecs)
    calculate_lambda(sim_results, "mangal_necklace_lambda.txt")
 