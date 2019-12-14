"""run training"""
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from vivid.featureset.molecules import find_molecule, MoleculeFactory

from kaggle_days.dataset import generate_next_step_dataset
from kaggle_days.dataset import read_data
from kaggle_days.train import TrainComposer


def parse_argument():
    parser = ArgumentParser(description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--molecule', default='benchmark',
                        choices=[str(m.name) for m in MoleculeFactory.molecules],
                        help='molecule name (see kaggle_days.molecules.py file)')
    parser.add_argument('--simple', action='store_true',
                        help='If True, run on small models (LightGBMx3 different objective function)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_argument()
    test_df = read_data(test=True)

    m = find_molecule(args.molecule)[0]
    pred_in_best = None
    for i in range(5):
        train_df, y = generate_next_step_dataset(y_pred=pred_in_best)
        composer = TrainComposer(molecule=m, simple=args.simple, suffix=f'psudo_{i}')
        score_df, pred_dict = composer.fit(train_df, y, test_df)

        best_model = score_df.sort_values('rmse').index[0]
        pred_in_best = pred_dict.get(best_model)
