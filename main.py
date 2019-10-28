import csv
import os
import pyrqa as rqa
from pyrqa.time_series import TimeSeries
from pyrqa.settings import Settings
from pyrqa.computing_type import ComputingType
from pyrqa.neighbourhood import FixedRadius
from pyrqa.metric import EuclideanMetric
from pyrqa.computation import RQAComputation
from pyrqa.computation import RPComputation
from pyrqa.image_generator import ImageGenerator
from collections import defaultdict

from praatclasses import TextGrid
from transitions.extensions import GraphMachine as Machine

VP_WORDS_PATH = os.path.join('VPs', 'Words')
VP_BLICKRICHTUNGEN_PATH = os.path.join('VPs', 'Blickrichtungen')
ANALYSEN_PATH = 'Analysen'


def analyze_eye_movement_patterns(interval_tier):
    pattern_dict = {}
    for n in range(0, 10):
        pattern_dict[n] = defaultdict((int))

    current_direction = None

    for interval in interval_tier:
        if current_direction is None:
            current_direction = int(interval.mark())
        else:
            next_direction = int(interval.mark())

            pattern_dict[current_direction][next_direction] += 1

            current_direction = next_direction

    return pattern_dict


def compute_relative_frequencies(pattern_dict):
    for blickrichtung in pattern_dict:
        sum = 0
        for k, v in pattern_dict[blickrichtung].items():
            sum += v

        for key in pattern_dict[blickrichtung]:
            value = pattern_dict[blickrichtung][key]
            pattern_dict[blickrichtung][key] = round(value / sum, 2)

    return pattern_dict


def write_movementpattern_to_csv(filename, pattern_dict):
    with open(filename, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([None] + [str(x) for x in range(0, 10)])
        for blickrichtung in pattern_dict:
            row = [str(pattern_dict[blickrichtung][x]) for x in range(0, 10)]
            writer.writerow([str(blickrichtung)] + row)


def create_transition_graph_from_dict(pattern_dict):
    machine = Machine(states=[str(x) for x in pattern_dict.keys()], use_pygraphviz=False)

    for blickrichtung in pattern_dict:
        for next_blckrchtng in pattern_dict[blickrichtung]:

            if pattern_dict[blickrichtung][next_blckrchtng] == 0:
                continue

            machine.add_transition(str(pattern_dict[blickrichtung][next_blckrchtng]), str(blickrichtung),
                                   str(next_blckrchtng))

    return machine


def create_recurrence_plot_from_intervaltier(interval_tier):
    data_points = [interval.mark() for interval in interval_tier]
    time_series = TimeSeries(data_points, embedding_dimension=2, time_delay=0)
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

    computation = RPComputation.create(settings)
    result = computation.run()
    ImageGenerator.save_recurrence_plot(result.recurrence_matrix_reverse,
                                        'recurrence_plot.png')

if __name__ == '__main__':
    # textgrid = TextGrid()
    # textgrid.read("2_vp21_words.TextGrid")
    # print(textgrid)

    textgrid2 = TextGrid()
    textgrid2.read(os.path.join(VP_BLICKRICHTUNGEN_PATH, '2_vp21_Blickrichtung.TextGrid'))
    print(textgrid2[0])
    textgrid2[0].delete_empty()
    print(textgrid2[0])
    pattern_dict = analyze_eye_movement_patterns(textgrid2[0])
    print(pattern_dict)
    print(compute_relative_frequencies(pattern_dict))
    write_movementpattern_to_csv(os.path.join(ANALYSEN_PATH, 'Blickrichtungsanalyse'), pattern_dict)
    machine = create_transition_graph_from_dict(pattern_dict)
    machine.get_combined_graph().draw("Blickrichtungsgraph.png")
    create_recurrence_plot_from_intervaltier(textgrid2[0])