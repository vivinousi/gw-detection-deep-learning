# EXAMPLE ONLY
# the following lines deploy the model on a foreground and background file respectively and save the resulting events in the results directory
mkdir results
# similarly, to run the improved model change test_challenge_model.py to test.py
CUDA_VISIBLE_DEVICES="0" python3 test_challenge_model.py dataset-4/v2/test_foreground_s24w6d1_1.hdf results/test_fgevents_s24w6d1_1.hdf
CUDA_VISIBLE_DEVICES="0" python3 test_challenge_model.py dataset-4/v2/test_background_s24w6d1_1.hdf results/test_bgevents_s24w6d1_1.hdf
# these scripts are part of the MLGWSC repository (https://github.com/gwastro/ml-mock-data-challenge-1 commit SHA: 2a3b8e7c6927b0d39a552bb3f1f4d615505bf6e9)
./evaluate.py --injection-file dataset-4/v2/test_injections_s24w6d1_1.hdf --foreground-events results/test_fgevents_s24w6d1_1.hdf --foreground-files dataset-4/v2/test_foreground_s24w6d1_1.hdf --background-events results/test_bgevents_s24w6d1_1.hdf --output-file results/test_eval_output_s24w6d1_1.hdf --verbose
python3 contributions/sensitivity_plot.py --files results/test_eval_output_s24w6d1_1.hdf --output results/test_eval_output_s24w6d1_1_plot.png --no-tex