"""proFit main script.

This script is called when running the `profit` command inside a shell.
"""

import sys
import os
from argparse import ArgumentParser
from platform import python_version

from profit.config import BaseConfig
from profit.util import safe_path
from profit.util.variable import VariableGroup, Variable
from profit.defaults import (
    base_dir as default_base_dir,
    config_file as default_config_file,
)
from profit.run import Runner


def main():
    """Main command line interface"""
    from profit import __version__  # delayed to prevent cyclic import

    # Get parameters from shell input
    # display help by default: https://stackoverflow.com/questions/4042452/display-help-message-with-python-argparse-when-script-is-called-without-any-argu
    class MyParser(ArgumentParser):
        def error(self, message):
            sys.stderr.write(f"error: {message}\n")
            self.print_help(sys.stderr)
            self.exit(2)

    parser = MyParser(
        description=f"Probabilistic Response Model Fitting with Interactive Tools v{__version__}"
    )
    subparsers = parser.add_subparsers(metavar="mode", dest="mode", required=True)
    subparsers.add_parser("run", help="start simulation runs")
    subparsers.add_parser("fit", help="fit data (e.g. with a Gaussian Process)")
    subparsers.add_parser("ui", help="interactively visualize results using dash")
    subparsers.choices["ui"].add_argument(
        "--debug", action="store_true", help="activate debug mode for the Dash app"
    )
    subparsers.add_parser(
        "clean", help="remove temporary files, run directories and logs"
    )
    subparsers.choices["clean"].add_argument(
        "--all", action="store_true", help="remove input, output and model files"
    )
    subparsers.add_parser("version", help="show version information")

    parser.add_argument(
        "base_dir",
        metavar="base-dir",
        help="path to config file or directory containing profit.yaml (default: current working directory)",
        default=default_base_dir,
        nargs="?",
    )

    args, reminder = parser.parse_known_args()

    # parse additional arguments which weren't used by the subparser by a modified main parser again
    parser.usage = parser.format_usage()
    del parser._actions[1]  # without subparsers
    args = parser.parse_args(reminder, namespace=args)

    # `profit version` does not require a config
    if args.mode == "version":
        print(f"proFit {__version__}")
        import profit.sur.gp.backend

        try:
            from profit.sur.gp.backend import gpfunc

            print("with fortran backend")
        except ImportError:
            print("without fortran backend")
        print(f"python {python_version()}")
        return

    # Instantiate Config from the given file
    config_file = safe_path(args.base_dir, default=default_config_file)
    config = BaseConfig.from_file(config_file)
    variables = config.variable_group

    sys.path.append(config["base_dir"])
    os.chdir(config["base_dir"])

    # Select mode
    if args.mode == "run":
        from tqdm import tqdm
        from profit.util import check_ndim
        from profit.util.file_handler import FileHandler

        runner = Runner.from_config(config["run"], config)  # Instantiate the runner
        if config["active_learning"]["resume_from"] is not None:
            X = FileHandler.load(config["files"]["input"])
            y = FileHandler.load(config["files"]["output"])
            for v in variables.list:
                if v.name in X.dtype.names:
                    v.value[: X.shape[0]] = X[v.name]
                else:
                    v.value[: y.shape[0]] = y[v.name]
        FileHandler.save(
            config["files"]["input"], variables.named_input
        )  # Save variables to input file

        # Check if active learning needs to be done instead of executing a normal run
        if "activelearning" in (v.kind.lower() for v in variables.list):
            from profit.al.active_learning import ActiveLearning
            from profit.sur.sur import Surrogate

            runner.fill(variables.named_input)  # Prepare runner with input variables
            al = ActiveLearning.from_config(
                runner, variables, config["active_learning"], config
            )  # Instantiate the active learning algorithm
            try:
                save_intermediate = {
                    "model_path": config["fit"]["save"] or config["fit"]["load"],
                    "input_path": config["files"]["input"],
                    "output_path": config["files"]["output"],
                }
                if config["active_learning"]["resume_from"]:
                    resume_from = config["active_learning"]["resume_from"]
                    runner.fill_output(variables.named_output)
                    runner.next_run_id = resume_from
                    al.learn(
                        resume_from=resume_from, save_intermediate=save_intermediate
                    )
                else:
                    al.warmup(
                        save_intermediate=save_intermediate
                    )  # Execute warmup cycles
                    al.learn(
                        save_intermediate=save_intermediate
                    )  # Execute main learning loop
                FileHandler.save(
                    config["files"]["input"], variables.named_input
                )  # Save learned input variables
            finally:
                runner.cancel_all()  # Close all run processes
            if config["active_learning"]["algorithm"]["save"]:
                al.save(config["active_learning"]["algorithm"]["save"])  # Save AL model
        else:
            # Normal (parallel) run
            params_array = [
                row[0] for row in variables.named_input
            ]  # Structured array of input values
            try:
                runner.spawn_array(
                    params_array, wait=True, progress=True
                )  # submit runs
            except KeyboardInterrupt:
                runner.logger.info("Keyboard Interrupt")
            finally:
                runner.cancel_all()  # Close all run processes

                # Write runner output data into variables
                for key in runner.output_data.dtype.names:
                    variables[key].value = check_ndim(runner.output_data[key])

        if len(runner.failed):
            runner.logger.warning(f"{len(runner.failed)} runs failed")

        # Format output data for txt file and save
        formatted_output_data = (
            variables.formatted_output
            if config["files"]["output"].endswith(".txt")
            else variables.named_output
        )
        FileHandler.save(config["files"]["output"], formatted_output_data)

    elif args.mode == "fit":
        from numpy import arange, hstack, meshgrid
        from profit.util.file_handler import FileHandler
        from profit.sur.sur import Surrogate

        sur = Surrogate.from_config(
            config["fit"], config
        )  # Instantiate surrogate model

        # Train model if not yet trained
        if not sur.trained:
            x = FileHandler.load(config["files"]["input"])
            y = FileHandler.load(config["files"]["output"])
            x = hstack(
                [x[key] for key in x.dtype.names]
            )  # Convert structured to normal array
            y = hstack([y[key] for key in y.dtype.names])

            sur.train(x, y)

        if config["fit"]["save"]:
            sur.save_model(config["fit"]["save"])  # Save surrogate model
        if config["ui"]["plot"]:
            # Make a simple plot of data and surrogate model
            # TODO: Rename to 'simple_plot' and introduce more options
            try:
                # Get prediction range from input or infer from training data
                xpred = [
                    arange(minv, maxv, step)
                    for minv, maxv, step in config["ui"]["plot"].get("Xpred")
                ]
                xpred = hstack([xi.flatten().reshape(-1, 1) for xi in meshgrid(*xpred)])
            except AttributeError:
                xpred = None
            sur.plot(xpred, independent=config["independent"], show=True)

    elif args.mode == "ui":
        from profit.ui import init_app

        app = init_app(config)
        app.run_server(debug=args.debug)  # Start Dash server on localhost

    elif args.mode == "clean":
        from shutil import rmtree
        from os import path, remove

        run_dir = config["run_dir"]

        # Remove single run directories
        for krun in range(config["ntrain"]):
            single_run_dir = path.join(run_dir, f"run_{krun:03d}")
            if path.exists(single_run_dir):
                rmtree(single_run_dir)

        if args.all:
            # Remove input and output files
            if path.exists(config["files"]["input"]):
                remove(config["files"]["input"])
            if path.exists(config["files"]["output"]):
                remove(config["files"]["output"])
            if path.exists(config["fit"]["save"]):
                remove(config["fit"]["save"])

        # Cleanup runner
        runner = Runner.from_config(config["run"], config)
        runner.clean()


if __name__ == "__main__":
    main()
