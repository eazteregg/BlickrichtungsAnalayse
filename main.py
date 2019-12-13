import csv
import math
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
from PIL import Image, ImageDraw

VP_WORDS_PATH = os.path.join('VPs', 'Words')
VP_BLICKRICHTUNGEN_PATH = os.path.join('VPs', 'Blickrichtungen')
VP_THINKANSWER_PATH = os.path.join("VPs", "ThinkAnswer")
ANALYSEN_PATH = 'Analysen'
CSV_PATH = 'csv'
GRAPH_PATH = 'graphs'
RECURRENCE_PATH = 'recPlots'
# change this dictionary if you want to change how the gaze directions are translated into colors; for current setup see
# Colored_CodingGrid.png
NUMBER2COLOR = {0: (102, 102, 102), 1: (0, 204, 255), 2: (0, 0, 255), 3: (0, 0, 128), 4: (102, 255, 51), 5: (0, 255, 0),
                6: (0, 128, 0), 7: (255, 128, 128), 8: (255, 0, 0), 9: (128, 0, 0)}
TA2COLOR = {"T": (191,191,191), "A": (64, 64, 64) }


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
                if next_direction != current_direction:
                    pattern_dict[current_direction][next_direction] += 1

                current_direction = next_direction

            except ValueError:
                del interval
                continue

    return pattern_dict


def compute_relative_frequencies(pattern_dict, withFive=True):
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


def create_transition_graph_from_dict(pattern_dict, withFive=True):
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


def create_recurrence_plot_from_intervaltier(blickrichtung_tier, thinkanswer_tier, destination, withFive=True):

    def create_list_from_thinkanswer_tier(blickrichtung_tier, thinkanswer_tier):

        thinkanswer_list = []

        for blkrchtng in blickrichtung_tier:
            for ta in thinkanswer_tier:
                if blkrchtng.xmin() >= ta.xmin() and blkrchtng.xmin() <= ta.xmax():
                    thinkanswer_list.append((blkrchtng.mark(), ta.mark()))
                    break
            else:
                for ta in thinkanswer_tier:
                    if math.floor(blkrchtng.xmin()) >= math.floor(ta.xmin()) and blkrchtng.xmin() <= ta.xmax():
                        thinkanswer_list.append((blkrchtng.mark(), ta.mark()))
                        break
                    elif round(blkrchtng.xmin()) >= round(ta.xmin()) and blkrchtng.xmin() <= ta.xmax():
                        thinkanswer_list.append((blkrchtng.mark(), ta.mark()))
                        break
                else:
                    print("No fit:" + str(blkrchtng))

        return thinkanswer_list

    thinkanswer_list = create_list_from_thinkanswer_tier(blickrichtung_tier, thinkanswer_tier)
    print("len :" + str(count_TAs(thinkanswer_list)))
    # turn interval tier into list of marks
    data_points = [interval.mark() for interval in blickrichtung_tier]

    print(len(data_points), " ", len (thinkanswer_list))

    # remove all fives if desired
    if not withFive:
        data_points = [mark for mark in data_points if mark != '5']

    data_points_clean = remove_doubles_from_list(data_points)




    time_series = TimeSeries(data_points_clean, embedding_dimension=1, time_delay=0)
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
    with open(destination + "_recAnal.txt", mode='w') as file:
        file.write(str(result))

    computation = RPComputation.create(settings)
    result = computation.run()
    ImageGenerator.save_recurrence_plot(result.recurrence_matrix_reverse,
                                        destination + "_recPlot.png")


    add_numbers_to_recurrence_plot(thinkanswer_list, destination + "_recPlot.png")


def remove_doubles_from_list(data_points, func= lambda x: x):

    # this elaborate code is designed to remove all subsequently recurring numbers in the gaze directions.
    # needed because during the encoding process, two subsequent 5s were sometimes placed right next to each other
    dic = dict()
    to_remove = []
    # enumerate all data points
    for n in range(len(data_points)):
        dic[n] = data_points[n]
    for key in dic:
        # can't remove the first one, so skip
        if key == 0:
            continue
        # if the two values next to each other are the same, ad done of them to the 2remove list
        if func(dic[key]) == func(dic[key - 1]):
            to_remove.append(key)
    # remove all previously identified values
    for n in to_remove:
        del dic[n]
    return list(dic.values())


def cleanup_IntervalTier(intervals):

    # change interval marks to literals where applicable
    for interval in intervals:
        if len(interval.mark()) > 1:
            interval.change_text(interval.mark()[0])


def add_numbers_to_recurrence_plot(numbers, recPlot, withQuestions=True):

    # open recurrence plot
    plotIm = Image.open(recPlot)

    if not withQuestions:
        horiOffset = 1
    else:
        horiOffset = 2
    # create new image, but 1px or 2px wider and 1px higher, depending on withQuestions, with backgroundcolor white
    newPlotIm = Image.new(plotIm.mode, (plotIm.width + horiOffset, plotIm.height + 1), color='white')
    # paste the opened recPlot into the new image at location (1,0) so that there remains a white strip to the left and
    # at the bottom of the new image
    newPlotIm.paste(plotIm, box=(horiOffset, 0))

    # create a dictionary to store the colors representing the numbers in; for reference see Colored_CodingGrid.png
    number_color_dict = dict()

    # fill in the colors corresponding to the numbers as specified in NUMBER2COLOR
    for number in NUMBER2COLOR:
        number_color_dict[NUMBER2COLOR[number]] = []


    print("len: " + str(count_TAs(numbers)))
    numbers_clean = remove_doubles_from_list(numbers, lambda x: x[0])
    print("len: " + str(count_TAs(numbers)))

    if withQuestions:
        question_color_dict = dict()
        for ta in TA2COLOR:
            question_color_dict[TA2COLOR[ta]] = []
    # fill in the dictionary with points at which to draw a particular color
    for number in range(len(numbers_clean)):
        number_color_dict[NUMBER2COLOR[int(numbers_clean[number][0])]] += [(horiOffset-1, (newPlotIm.height-2) - (number)), (number+horiOffset, newPlotIm.height-1)]

        if withQuestions:
            question_color_dict[TA2COLOR[numbers_clean[number][1][0]]] += [(0, (newPlotIm.height-2) - number)]

    plotDraw = ImageDraw.Draw(newPlotIm)

    for color in number_color_dict:
         plotDraw.point(number_color_dict[color], color)

    for color in question_color_dict:
        plotDraw.point(question_color_dict[color], color)


    newPlotIm.save(recPlot[:-4] + "_numbered.png")

def do_Analysis(withFive=True):
    regex = r'(\d*_*vp\d*)_.*\.TextGrid'

    # this reads in the Blickrichtungen tiers from all participants
    VP_brGrids = dict()

    for filename in os.listdir(VP_BLICKRICHTUNGEN_PATH):
        mo = re.search(regex, filename)
        if mo:
            vp_nr = mo.group(1)
            VP_brGrids[vp_nr] = TextGrid()
            VP_brGrids[vp_nr].read(os.path.join(VP_BLICKRICHTUNGEN_PATH, filename))
            VP_brGrids[vp_nr][0].delete_empty()
            cleanup_IntervalTier(VP_brGrids[vp_nr][0])

    # this reads in the ThinkAnswer tiers from all participants; needed for the recurrence plots
    VP_taGrids = dict()
    for filename in os.listdir(VP_THINKANSWER_PATH):
        mo = re.search(regex, filename)
        if mo:
            vp_nr = mo.group(1)
            VP_taGrids[vp_nr] = TextGrid()
            VP_taGrids[vp_nr].read(os.path.join(VP_THINKANSWER_PATH, filename))
            VP_taGrids[vp_nr][0].delete_empty()

    VP_PatternDicts = dict()

    for vp_nr in VP_brGrids:
        VP_PatternDicts[vp_nr] = analyze_eye_movement_patterns(VP_brGrids[vp_nr][0])

        compute_relative_frequencies(VP_PatternDicts[vp_nr], withFive)

    for vp_nr in VP_PatternDicts:
        write_movementpattern_to_csv(os.path.join(ANALYSEN_PATH, CSV_PATH, vp_nr + "_tabelle.csv"),
                                     VP_PatternDicts[vp_nr])

        machine = create_transition_graph_from_dict(VP_PatternDicts[vp_nr], withFive)
        machine.get_combined_graph().draw(os.path.join(ANALYSEN_PATH, GRAPH_PATH, vp_nr + "_graph.png"))

    for vp_nr in VP_brGrids:
        create_recurrence_plot_from_intervaltier(VP_brGrids[vp_nr][0], VP_taGrids[vp_nr][0], os.path.join(ANALYSEN_PATH, RECURRENCE_PATH,
                                                                                      vp_nr), withFive)

def count_TAs(lis):
        return len(set(x[1] for x in lis))

if __name__ == '__main__':
    do_Analysis(withFive=True)
