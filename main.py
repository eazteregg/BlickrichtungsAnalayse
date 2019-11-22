import csv
import os
from pyrqa.time_series import TimeSeries
from pyrqa.settings import Settings
from pyrqa.computing_type import ComputingType
from pyrqa.neighbourhood import FixedRadius
from pyrqa.metric import EuclideanMetric
from pyrqa.computation import RQAComputation
from pyrqa.computation import RPComputation
from pyrqa.image_generator import ImageGenerator
from collections import defaultdict
import re
from praatclasses import TextGrid
from transitions.extensions import GraphMachine as Machine

VP_WORDS_PATH = os.path.join('VPs', 'Words')
VP_BLICKRICHTUNGEN_PATH = os.path.join('VPs', 'Blickrichtungen')
ANALYSEN_PATH = 'Analysen'
CSV_PATH = 'csv'
GRAPH_PATH = 'graphs'
RECURRENCE_PATH = 'recPlots'


def analyze_eye_movement_patterns(interval_tier):

    pattern_dict = {}
    for n in range(0, 10):
        pattern_dict[n] = defaultdict((int))

    current_direction = None

    for interval in interval_tier:

        if current_direction is None:
            try:
                current_direction = int(interval.mark())
            except ValueError:
                del interval
                continue
        else:
            try:
                next_direction = int(interval.mark())
                pattern_dict[current_direction][next_direction] += 1

                current_direction = next_direction

            except ValueError:
                del interval
                continue



    return pattern_dict


def compute_relative_frequencies(pattern_dict, withFive = True):
    for blickrichtung in pattern_dict:
        sum = 0
        for key, v in pattern_dict[blickrichtung].items():
            if not withFive and key == 5:
                continue
            sum += v

        for key in pattern_dict[blickrichtung]:
            if not withFive and key == 5:
                continue
            value = pattern_dict[blickrichtung][key]
            pattern_dict[blickrichtung][key] = round(value / sum, 2)

    return pattern_dict


def write_movementpattern_to_csv(filename, pattern_dict, withFive=True):
    with open(filename, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([None] + [str(x) for x in range(0, 10)])
        for blickrichtung in pattern_dict:

            if not withFive and blickrichtung == 5:
                continue
            row = [str(pattern_dict[blickrichtung][x]) for x in range(0, 10)]
            writer.writerow([str(blickrichtung)] + row)


def create_transition_graph_from_dict(pattern_dict, withFive = True):
    if withFive:
        machine = Machine(states=[str(x) for x in pattern_dict.keys()], use_pygraphviz=False)

    else:
        machine = Machine(states=[str(x) for x in pattern_dict.keys() if x != 5], use_pygraphviz=False)

    for blickrichtung in pattern_dict:
        for next_blckrchtng in pattern_dict[blickrichtung]:

            if not withFive and (blickrichtung == 5 or next_blckrchtng == 5):
                continue
            if pattern_dict[blickrichtung][next_blckrchtng] == 0:
                continue


            machine.add_transition(str(pattern_dict[blickrichtung][next_blckrchtng]), str(blickrichtung),
                                   str(next_blckrchtng))

    return machine


def create_recurrence_plot_from_intervaltier(interval_tier, destination, withFive=True):
    data_points = [interval.mark() for interval in interval_tier]

    if not withFive:
        for mark in data_points:
            if mark == 5:
                del mark

    time_series = TimeSeries(data_points, embedding_dimension=1, time_delay=0)
    settings = Settings(time_series,
                        computing_type=ComputingType.Classic,
                        neighbourhood=FixedRadius(0.65),
                        similarity_measure=EuclideanMetric,
                        theiler_corrector=1)

    computation = RQAComputation.create(settings,
                                        verbose=True)
    result = computation.run()
    result.min_diagonal_line_length = 2
    result.min_vertical_line_length = 2
    result.min_white_vertical_line_lelngth = 2
    print(result)
    with open(destination + "_recAnal.txt", mode='w') as file:
        file.write(str(result))

    computation = RPComputation.create(settings)
    result = computation.run()
    ImageGenerator.save_recurrence_plot(result.recurrence_matrix_reverse,
                                        destination + "_recPlot.png")

def cleanup_IntervalTier(intervals):


    # change interval marks to literals where applicable
    for interval in intervals:
        if len(interval.mark()) > 1:
            interval.change_text(interval.mark()[0])

    intervals.delete_doubles()

def do_Analysis(withFive = True):
    regex = r'(\d*_*vp\d*)_.*\.TextGrid'

    VP_TextGrids = dict()

    for filename in os.listdir(VP_BLICKRICHTUNGEN_PATH):
        mo = re.search(regex, filename)
        if mo:
            vp_nr = mo.group(1)
            VP_TextGrids[vp_nr] = TextGrid()
            VP_TextGrids[vp_nr].read(os.path.join(VP_BLICKRICHTUNGEN_PATH, filename))
            VP_TextGrids[vp_nr][0].delete_empty()
            cleanup_IntervalTier(VP_TextGrids[vp_nr][0])

    VP_PatternDicts = dict()

    for vp_nr in VP_TextGrids:
        VP_PatternDicts[vp_nr] = analyze_eye_movement_patterns(VP_TextGrids[vp_nr][0])

        compute_relative_frequencies(VP_PatternDicts[vp_nr], withFive)

    for vp_nr in VP_PatternDicts:
        write_movementpattern_to_csv(os.path.join(ANALYSEN_PATH, CSV_PATH, vp_nr + "_tabelle.csv"),
                                     VP_PatternDicts[vp_nr])

        machine = create_transition_graph_from_dict(VP_PatternDicts[vp_nr], withFive)
        machine.get_combined_graph().draw(os.path.join(ANALYSEN_PATH, GRAPH_PATH, vp_nr + "_graph.png"))

    for vp_nr in VP_TextGrids:
        create_recurrence_plot_from_intervaltier(VP_TextGrids[vp_nr][0], os.path.join(ANALYSEN_PATH, RECURRENCE_PATH,
                                                                                         vp_nr), withFive)


if __name__ == '__main__':

    do_Analysis(withFive=True)