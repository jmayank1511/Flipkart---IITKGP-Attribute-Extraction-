# Given input train and test files, generates combined train and validation and
# test files by merging and then shuffling the contents.
from sklearn.utils import shuffle

class dataMergeShuffler(object):

    def alternate(self, list1, list2):
        small_list = list1 if len(list1) < len(list2) else list2
        large_list = list1 if len(list1) > len(list2) else list2
        results = []
        for i in range(0, len(small_list)):
            results.append(small_list[i])
            results.append(large_list[i])

        for i in range(len(small_list), len(large_list)):
            results.append(large_list[i])

        return results

    def augment_file(self, vertical, fname, _id=None, outfname=None):
        results = []
        curr = []
        for line in open(fname):
            line = line.strip()
            if not line:
                results.append(curr[:])
                curr = []
                continue
            word, tag = line.strip().split(" ")
            tag = "{}_{}".format(tag, vertical) if tag != "O" else tag
            if _id:
                curr.append((word, tag, str(_id)))
            else:
                curr.append((word, tag))

        if curr:
            results.append(curr[:])
        
        if outfname:
            with open(outfname, "w") as fw:
                for desc in results:
                    for tup in desc:
                        fw.write(" ".join(tup))
                        fw.write("\n")
                    fw.write("\n")
        
        return results

    def merge_shufflie_files(self, v1, v1_file, v2, v2_file, augument=True, outfname=None, alternate=False):
        if augument:
            v1_results = self.augment_file(v1, v1_file, 1)
            v2_results = self.augment_file(v2, v2_file, 2)
        else:
            v1_results = self.augment_file(v1, v1_file)
            v2_results = self.augment_file(v2, v2_file)

        if alternate:
            tot_results = self.alternate(v1_results, v2_results)
        else:
            tot_results = shuffle(v1_results + v2_results, random_state=16)
        
        if outfname:
            with open(outfname, "w") as fw:
                for desc in tot_results:
                    for tup in desc:
                        fw.write(" ".join(tup))
                        fw.write("\n")
                    fw.write("\n")

        return tot_results

if __name__ == "__main__":
    v1 = "dress"
    v2 = "jean"
    v1_train = "dress_small_train.txt"
    v2_train = "jean_small_train.txt"
    v1_test = "dress_small_test.txt"
    v2_test = "jean_small_test.txt"
    v1_test_out = "dress_a_small_test.txt"
    v2_test_out = "jean_a_small_test.txt"
    train_out = "dress_jean_a_train.txt"
    val_out = "dress_jean_a_val.txt"

    merger = dataMergeShuffler()
    merger.augment_file(v1, v1_test, 1, v1_test_out)
    merger.augment_file(v2, v2_test, 2, v2_test_out)
    merger.merge_shufflie_files(v1, v1_train, v2, v2_train, True, train_out, True)
    merger.merge_shufflie_files(v1, v1_test, v2, v2_test, True, val_out, True)

