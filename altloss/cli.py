import sys
import json
import argparse as ap
import cmdstanpy as stan
import altloss.mrp as mrp
from importlib import resources as res
from pathlib import Path


def parse_arguments():
    parser = ap.ArgumentParser()
    subparsers = parser.add_subparsers(help="subcommand help")
    # setup subcommand
    parser_setup = subparsers.add_parser("setup",
        help = "download and compile cmdstan"
    )
    parser_setup.set_defaults(func=setup_cmdstan)
    # fit subcommand
    parser_fit = subparsers.add_parser("fit",
        help = "fit model to acoustic data"
    )
    parser_fit.add_argument("-O", "--obs",
        help = "observation file",
        required=True,
        type = Path
    )
    parser_fit.add_argument("-T", "--tags",
        help = "tag data file",
        required=True,
        type = Path
    )
    parser_fit.add_argument("-S", "--species",
        help = "species of interest",
        required=True,
        type = str
    )
    parser_fit.add_argument("-o", "--output",
        help = "model fit output",
        default = Path('output.csv'),
        type = Path
    )
    parser_fit.add_argument("-d", "--debug",
        help = "directory to write output files to (for debugging)",
        type = Path
    )
    parser_fit.add_argument('-s', '--summary',
        help = "inference summary file",
        type = Path
    )
    parser_fit.add_argument('--no-sim',
        help = "do not perform simulation (only inference)",
        action = 'store_true'
    )
    parser_fit.add_argument("-B", "--batch-size",
        help = "how many fish to simulate in each batch",
        type = int,
        default = 1000
    )
    parser_fit.add_argument("-R", "--num_batches",
        help = "how many batches to simulate",
        type = int,
        default = 1000
    )
    parser_fit.set_defaults(func=mrp_fit)

    # parse and return the args
    args = parser.parse_args()
    return args


def setup_cmdstan(args):
    stan.install_cmdstan()


def mrp_fit(args):
    with res.path('altloss.stan', 'mrp3.stan') as stan_file:
        print(f'Compiling Stan model {stan_file}', file=sys.stderr)
        model = stan.CmdStanModel(model_name='mrp', stan_file=stan_file)
    print("loading data", file=sys.stderr)
    stan_data = mrp.load_stan_data(args.obs, args.tags, {args.species})
    fit = model.sample(data=stan_data, output_dir=args.debug)
    if not (args.summary is None):
        fit.summary().to_csv(args.summary, header=True, index=True)
        print("inference summary written to {args.summary}", file=sys.stderr)
    if args.no_sim:
        print("skipping simulation", file=sys.stderr)
        return
    print("starting simulation stage", file=sys.stderr)
    df = mrp.run_batches(stan_data, fit, batch = args.batch_size, reps = args.num_batches)
    print(f'writing simulation results to {args.output}', file=sys.stderr)
    df.to_csv(args.output, header=True, index=False)


def main():
    args = parse_arguments()
    print(args, file=sys.stderr)
    args.func(args)
