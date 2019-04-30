"""
This module reads a file which contains entries of format <word actual_tag pred_tag>
"""
from __future__ import division
import sys
from collections import namedtuple
from copy import deepcopy

Entity = namedtuple("Entity", "tag start end")


def check_overlap(entity_1, entity_2):
    """
    Check if two entities overlap.
    """
    return (entity_1.start <= entity_2.end) and (entity_2.start <= entity_1.end)


def check_exact_boundary_match(entity_1, entity_2):
    return (entity_1.start == entity_2.start) and (entity_1.end == entity_2.end)


def check_partial_entity_match(entity_1, entity_2):
    return check_overlap(entity_1, entity_2) and entity_1.start == entity_2.start


def register_entity(entity, entities, start_list, end_list):
    """
    A helper function to put entity into the list entities, and assign the entity
    to corresponding start and end indices in start_list and end_list
    """
    entities.append(entity)
    start_list[entity.start] = entity
    end_list[entity.end] = entity


def get_entities(tags):
    """
    From a given list of tokens it extracts entities. All consecutive B and i tags
    are combined to form one entity. Each entity is a tuple (tag, start_index,
    end_index)
    """
    entities = []
    start_dict = [None] * len(tags)
    end_dict = [None] * len(tags)
    tag, start, end = None, None, None

    for i, e in enumerate(tags):
        if e == "O":
            if tag is not None:
                end = i - 1
                register_entity(Entity(tag, start, end),
                                entities, start_dict, end_dict)
                tag, start, end = None, None, None
        else:
            if tag is None:
                tag = e[2:]
                start = i
                end = None
            elif tag != e[2:]:
                end = i - 1
                register_entity(Entity(tag, start, end), entities,
                                start_dict, end_dict)
                tag = e[2:]
                start = i
                end = None
    if tag and start and not end:
        register_entity(Entity(tag, start, i), entities, start_dict, end_dict)
    return entities, start_dict, end_dict


def run_evaluate_compound(actual_tags, predicted_tags, tagset):
    """
    Calculates precison and recall by calculating correct, incorrect, spurious
    and missing for each tag.
    """
    metrics = {"correct": 0, "incorrect": 0, "missing": 0, "spurious": 0}
    true_entities, true_start_list, true_end_list = get_entities(actual_tags)
    pred_entities, pred_start_list, pred_end_list = get_entities(predicted_tags)
    metrics_per_tag = {tag: deepcopy(metrics) for tag in tagset}
    # we use these two lists  to figure out what entities are spurious and
    # missing. Spurious entities are predicted entities which have no
    # corresponding true entities, and missing entities are true entities which
    # have no corresponding predicted entities.
    non_missing_true_entities = []
    non_spurious_pred_entities = []

    for true_entity in true_entities:

        if true_entity in pred_entities:
            metrics_per_tag[true_entity.tag]["correct"] += 1
            non_spurious_pred_entities.append(true_entity)
            non_missing_true_entities.append(true_entity)
            metrics["correct"] += 1

        else:
            # all possible entities that could overlap with the given entity
            # whose start is "i" and end is "j" are: all entities whose start is
            # less than "j" and end is greater than "i"
            start_entities = set(pred_start_list[:true_entity.end])
            end_entities = set(pred_end_list[true_entity.start:])
            overlapping_entities = start_entities.intersection(end_entities)
            overlapping_entities = [e for e in overlapping_entities
                                    if e is not None]
            for pred_entity in overlapping_entities:
                if check_overlap(pred_entity, true_entity):
                    metrics_per_tag[true_entity.tag]["incorrect"] += 1
                    metrics["incorrect"] += 1
                    non_spurious_pred_entities.append(pred_entity)
                    non_missing_true_entities.append(true_entity)
                    break

    for entity in pred_entities:
        if entity not in non_spurious_pred_entities:
            metrics_per_tag[entity.tag]["spurious"] += 1
            metrics["spurious"] += 1
    for entity in true_entities:
        if entity not in non_missing_true_entities:
            metrics_per_tag[entity.tag]["missing"] += 1
            metrics["missing"] += 1

    possible = metrics["correct"] + metrics["incorrect"] + metrics["missing"]
    actual = metrics["correct"] + metrics["incorrect"] + metrics["spurious"]
    p = metrics["correct"]/actual
    r =  metrics["correct"]/possible
    print "Precision:"+ " "+ str(p), "Recall:" + " " + str(r)
    print "F-1 Score:"+ " " + str(2/((1/p)+(1/r)))
    #print "Accuracy:" + " " + str(metrics["correct"] * 100 / (metrics["correct"] + metrics["incorrect"] + metrics["missing"] + metrics["spurious"]))
    print metrics
    return metrics_per_tag


if __name__ == "__main__":
    input_fname = sys.argv[1]
    actuals = []
    predictions = []
    possible_tags = set()

    with open(input_fname) as fp:
        for line in fp:
            line = line.strip()
            if line == "":
                continue
            word, actual, predicted = line.split()
            predictions.append(predicted)
            actuals.append(actual)
            if actual == "O":
                continue
            else:
                possible_tags.add(actual[2:])
    print possible_tags
    metrics = run_evaluate_compound(actuals, predictions, possible_tags)
    #print metrics
    f = open("results.txt",'w')
    f.write("Class" + "\t" + "Possible" + "\t" +"Correct" + "\t" + "Incorrect" + "\t"+ "Missing" + "\t" + "Spurious" + "\n" )
    for k,v in metrics.items():
        possible = v["correct"] + v["incorrect"] + v["missing"]
        #actual = v["correct"] + v["incorrect"] + v["spurious"]
        print k, possible, v["correct"], v["incorrect"], v["missing"], v["spurious"]
        f.write(k + "\t" + str(possible) + "\t" + str(v["correct"]) + "\t" + str(v["incorrect"]) + "\t" + str(v["missing"]) + "\t" + str(v["spurious"]) + "\n")
    f.close()

