import csv
import math
import os
import re
import json
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
NUMBER2COLOR = {0: (102, 102, 102), 1: (0, 204, 255), 2: (0, 0, 255), 3: (0, 0, 128), 4: (196, 252, 176), 5: (0, 255, 0),
                6: (0, 128, 0), 7: (255, 128, 128), 8: (255, 0, 0), 9: (128, 0, 0)}
TA2COLOR = {0: (191,191,191), 1: (64, 64, 64)}
CONDITION2COLOR = {"f": (255, 77, 255), "p" :(51, 204, 51), "s": (255, 153, 51)}

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
            for n in range(len(thinkanswer_tier)):
                mark = thinkanswer_tier[n].mark()
                if mark[0] == "T":
                    question_xmin = thinkanswer_tier[n].xmin()
                    question_xmax = thinkanswer_tier[n+1].xmax()
                    if question_xmin <= blkrchtng.xmin() <= question_xmax:
                        thinkanswer_list.append((blkrchtng.mark(), n//2, mark))
                        break
            else:
                for n in range(len(thinkanswer_tier)):
                    mark = thinkanswer_tier[n].mark()
                    if mark[0] == "T":
                        question_xmin = thinkanswer_tier[n].xmin()
                        question_xmax = thinkanswer_tier[n + 1].xmax()
                        if   math.floor(question_xmin) <= math.floor(blkrchtng.xmin()) <= question_xmax:
                            thinkanswer_list.append((blkrchtng.mark(), n // 2, mark))
                            break
                        elif  round(question_xmin) <= round(blkrchtng.xmin()) <= question_xmax:
                            thinkanswer_list.append((blkrchtng.mark(), n // 2, mark))
                            break
                else:
                    print("No fit:" + str(blkrchtng))

        return thinkanswer_list

    thinkanswer_list = create_list_from_thinkanswer_tier(blickrichtung_tier, thinkanswer_tier)
    print("len :" + str(count_TAs(thinkanswer_list)))
    # turn interval tier into list of marks

    print("len: " + str(count_TAs(thinkanswer_list)))
    thinkanswer_list_clean = remove_doubles_from_list(thinkanswer_list, lambda x: x[:2])
    print("len: " + str(count_TAs(thinkanswer_list)))

    data_points = [x[0] for x in thinkanswer_list_clean]


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
    with open(destination + "_recAnal.txt", mode='w') as file:
        file.write(str(result))

    computation = RPComputation.create(settings)
    rp_result = computation.run()
    ImageGenerator.save_recurrence_plot(rp_result.recurrence_matrix_reverse,
                                        destination + "_recPlot.png")


    add_numbers_to_recurrence_plot(thinkanswer_list_clean, destination + "_recPlot.png")

    return result

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
        if key == 0 or key-1 in to_remove:
            continue

        # if the two values next to each other are the same, add one of them to the 2remove list
        prev = func(dic[key-1])
        this = func(dic[key])
        try:
            nex = func(dic[key+1])
        except KeyError:
            nex = "False"
        if len(prev) > 1:
            if prev[0] == this[0]:
                if prev[1] == this[1] or this[1] == nex[1]:
                    to_remove.append(key)
        elif prev == this:
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
        horiOffset = 3
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

    if withQuestions:

        question_color_dict = dict()
        conditions_color_dict = dict()
        for ta in TA2COLOR:
            question_color_dict[TA2COLOR[ta]] = []
        for condition in CONDITION2COLOR:
            conditions_color_dict[CONDITION2COLOR[condition]] = []
    # fill in the dictionary with points at which to draw a particular color
    for number in range(len(numbers)):
        number_color_dict[NUMBER2COLOR[int(numbers[number][0])]] += [(horiOffset-1, (newPlotIm.height-2) - (number)), (number+horiOffset, newPlotIm.height-1)]

        if withQuestions:
            question_color_dict[TA2COLOR[numbers[number][1]%2]] += [(horiOffset -2, (newPlotIm.height-2) - number)]
            conditions_color_dict[CONDITION2COLOR[numbers[number][2][1]]] += [(horiOffset - 3, (newPlotIm.height-2) - number)]

    plotDraw = ImageDraw.Draw(newPlotIm)

    for color in number_color_dict:
         plotDraw.point(number_color_dict[color], color)

    for color in question_color_dict:
        plotDraw.point(question_color_dict[color], color)

    for color in conditions_color_dict:
        plotDraw.point(conditions_color_dict[color], color)

    newPlotIm.save(recPlot[:-4] + "_numbered.png")

def do_Analysis(withFive=True):
    regex = r'(\d*_*vp\d*)_.*\.TextGrid'

    # this reads in the Blickrichtungen (gaze directions) tiers from all participants
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

    # create the transition matrix for the gaze directions of each participant
    for vp_nr in VP_PatternDicts:
        # also save the transition matrix as a csv file just because
        write_movementpattern_to_csv(os.path.join(ANALYSEN_PATH, CSV_PATH, vp_nr + "_tabelle.csv"),
                                     VP_PatternDicts[vp_nr])

        machine = create_transition_graph_from_dict(VP_PatternDicts[vp_nr], withFive)
        machine.get_combined_graph().draw(os.path.join(ANALYSEN_PATH, GRAPH_PATH, vp_nr + "_graph.png"))

    # create the recurrence plots for each participant and save the rqa results in a list for later
    rqa_results = []
    for vp_nr in VP_brGrids:
         result = create_recurrence_plot_from_intervaltier(VP_brGrids[vp_nr][0], VP_taGrids[vp_nr][0], os.path.join(ANALYSEN_PATH, RECURRENCE_PATH,
                                                                                      vp_nr), withFive)
         rqa_results.append((vp_nr,result))

    # write results into a nice csv-table
    with open(os.path.join(ANALYSEN_PATH, "OverallRqaResults.csv"), 'w') as csvfile:
        fieldnames = None
        writer = None
        for vp_nr, result in rqa_results:
            print(to_json(result))
            result_dict = json.loads(to_json(result))
            if fieldnames == None:
                fieldnames = ['VP'] + list(result_dict.keys())
            if writer == None:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
            result_dict['VP'] = vp_nr
            writer.writerow(result_dict)

def count_TAs(lis):
        return len(set(x[1] for x in lis))

def to_json(rqa_result):
    return json.dumps({"Minimum diagonal line length (L_min)": float(rqa_result.min_diagonal_line_length),
                       "Minimum vertical line length (V_min)": float(rqa_result.min_vertical_line_length),
                       "Minimum white vertical line length (W_min)": float(rqa_result.min_white_vertical_line_length),
                       "Recurrence rate (RR)": float(rqa_result.recurrence_rate),
                       "Determinism (DET)": float(rqa_result.determinism),
                       "Average diagonal line length (L)": float(rqa_result.average_diagonal_line),
                       "Longest diagonal line length (L_max)": float(rqa_result.longest_diagonal_line),
                       "Divergence (DIV)": float(rqa_result.divergence),
                       "Entropy diagonal lines (L_entr)": float(rqa_result.entropy_diagonal_lines),
                       "Laminarity (LAM)": float(rqa_result.laminarity),
                       "Longest vertical line length (V_max)": float(rqa_result.longest_vertical_line),
                       "Entropy vertical lines (V_entr)": float(rqa_result.entropy_vertical_lines),
                       "Average white vertical line length (W)": float(rqa_result.average_white_vertical_line),
                       "Longest white vertical line length (W_max)": float(rqa_result.longest_white_vertical_line),
                       "Longest white vertical line length inverse (W_div)": float(rqa_result.longest_white_vertical_line_inverse),
                       "Entropy white vertical lines (W_entr)": float(rqa_result.entropy_white_vertical_lines),
                       "Ratio determinism / recurrence rate (DET/RR)": float(rqa_result.ratio_determinism_recurrence_rate),
                       "Ratio laminarity / determinism (LAM/DET)": float(rqa_result.ratio_laminarity_determinism)
                       },
                      sort_keys=False,
                      indent=4,
                      separators=(',', ': '))
if __name__ == '__main__':
    do_Analysis(withFive=True)
