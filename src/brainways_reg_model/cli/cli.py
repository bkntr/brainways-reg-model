import click

from brainways_reg_model.cli.prepare_synth_data import prepare_synth_data
from brainways_reg_model.model.train import train


@click.group()
def cli():
    pass


cli.add_command(prepare_synth_data, name="prepare-synth-data")
cli.add_command(train, name="train")


if __name__ == "__main__":
    cli()
