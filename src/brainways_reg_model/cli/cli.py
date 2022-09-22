import click

from brainways_reg_model.cli.prepare_real_data import prepare_real_data
from brainways_reg_model.cli.prepare_synth_data import prepare_synth_data
from brainways_reg_model.model.evaluate import evaluate
from brainways_reg_model.model.finetune import finetune
from brainways_reg_model.model.train import train


@click.group()
def cli():
    pass


cli.add_command(prepare_synth_data, name="prepare-synth-data")
cli.add_command(prepare_real_data, name="prepare-real-data")
cli.add_command(train, name="train")
cli.add_command(finetune, name="finetune")
cli.add_command(evaluate, name="eval-finetune")


if __name__ == "__main__":
    cli()
