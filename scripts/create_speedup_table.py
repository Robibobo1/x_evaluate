import argparse
import os

import numpy as np
import pandas as pd

from x_evaluate.evaluation_data import EvaluationDataSummary
from x_evaluate.scriptlets import read_evaluation_pickle
from x_evaluate.trajectory_evaluation import create_trajectory_result_table_wrt_traveled_dist
from x_evaluate.utils import get_common_stats_functions, merge_tables


def create_rtf_table(s: EvaluationDataSummary):

    datasets = s.data.keys()

    data = np.empty((len(datasets), 4), dtype=float)
    i = 0
    for k in s.data.keys():
        stats_func = get_common_stats_functions()
        rtfs = s.data[k].performance_data.rt_factors

        data[i, :] = [stats_func['min'](rtfs), stats_func['median'](rtfs),
                      stats_func['max'](rtfs), stats_func['mean'](rtfs)]
        i += 1

    index_columns = [(s.name, "RTF Min"), (s.name, "RTF Median"),
                     (s.name, "RTF Max"), (s.name, "RTF Mean")]
    index = pd.MultiIndex.from_tuples(index_columns, names=["Evaluation Run", "Metric"])
    result_table = pd.DataFrame(data, index=datasets, columns=index)

    return result_table


def main():
    parser = argparse.ArgumentParser(description='Speedup through event filtering analysis')
    parser.add_argument('--input_file', type=str, required=True, help='Full path to evaluation file')

    args = parser.parse_args()

    evaluation = read_evaluation_pickle(os.path.dirname(args.input_file))

    merge_with_new_eval_run(evaluation, '/storage/data/projects/nasa-eve/experiments/speedup/local/2022-05-08-poster-6dof-speedup/000-speedup-run')
    merge_with_new_eval_run(evaluation, '/storage/data/projects/nasa-eve/experiments/speedup/local/2022-05-09-poster'
                                        '-6dof-speedup/000-speedup-run-konstantin')
    merge_with_new_eval_run(evaluation, '/storage/data/projects/nasa-eve/experiments/speedup/local/2022-05-10-poster'
                                        '-6dof-speedup/000-speedup-run-konstantin')
    merge_with_new_eval_run(evaluation, '/storage/data/projects/nasa-eve/experiments/speedup/local/2022-05-11-poster'
                                        '-6dof-speedup/000-speedup-run-konstantin')

    pd.options.display.max_colwidth = None
    pd.options.display.width = 0

    error_table = create_trajectory_result_table_wrt_traveled_dist(evaluation)
    rtf_table = create_rtf_table(evaluation)

    table = merge_tables([error_table, rtf_table])

    cropped_table = table[[('Poster 6DOF Speedup run', 'Mean Position Error [%]'), ('Poster 6DOF Speedup run',
                                                                                    'Mean Yaw error [deg/m]'),
                           ('Poster 6DOF Speedup run', 'RTF Max'), ('Poster 6DOF Speedup run', 'RTF Mean'),
                           ('Poster 6DOF Speedup run', 'RTF Median')]]
    print()
    print(cropped_table.round(2))
    cropped_table.round(2).to_excel('speedup.xlsx')


def merge_with_new_eval_run(evaluation, other_name):
    evaluation_2 = read_evaluation_pickle(other_name)
    keys_to_copy = list(evaluation_2.data.keys())
    for k in keys_to_copy:
        evaluation.data[k] = evaluation_2.data[k]


if __name__ == '__main__':
    main()
